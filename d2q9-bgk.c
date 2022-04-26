/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "mpi.h"

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"
#define NROWS 4
#define NCOLS 16
#define MASTER 0

/* struct to hold the parameter values */
typedef struct
{
  int nx;           /* no. of cells in x-direction */
  int ny;           /* no. of cells in y-direction */
  int maxIters;     /* no. of iterations */
  int reynolds_dim; /* dimension for Reynolds number */
  float density;    /* density per link */
  float accel;      /* density redistribution */
  float omega;      /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float* restrict speed_0;
  float* restrict speed_1;
  float* restrict speed_2;
  float* restrict speed_3;
  float* restrict speed_4;
  float* restrict speed_5;
  float* restrict speed_6;
  float* restrict speed_7;
  float* restrict speed_8;

} t_speed;


/*
** function prototypes
*/

/* function prototypes */
int calc_ncols_from_rank(int rank, int size);

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
               unsigned int **obstacles_ptr, float **av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, unsigned int* restrict obstacles);
int accelerate_flow(const t_param params, t_speed* restrict cells, unsigned int* restrict obstacles);
int write_values(const t_param params, t_speed* restrict cells, unsigned int* restrict obstacles, float* restrict av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, t_speed** restrict cells_ptr, t_speed** restrict tmp_cells_ptr,
             unsigned int** restrict obstacles_ptr, float** restrict av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* restrict cells);


/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* restrict cells, unsigned int* restrict obstacles);

/* utility functions */
void die(const char *message, const int line, const char *file);
void usage(const char *exe);

/* global variable */
unsigned int is_power_of_2;

unsigned int check_power_of_2(unsigned int x) {
  unsigned int pow = 0;
  unsigned int result = x;
  while (result != 1)
  {
    result = result >> 1;
    if ((result+1) & 1) return 0; 
    pow++;
  }
  return 1;
}

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char *argv[])
{
  //MPI INIT
  int flag;               /* for checking whether MPI_Init() has been called */
  int strlen_;             /* length of a character array */
  enum bool {FALSE,TRUE}; /* enumerated type: false = 0, true = 1 */  
  char hostname[MPI_MAX_PROCESSOR_NAME];  /* character array to hold hostname running process */
  int ii,jj;             /* row and column indices for the grid */
  int kk;                /* index for looping over ranks */
  int rank;              /* the rank of this process */
  int left;              /* the rank of the process to the left */
  int right;             /* the rank of the process to the right */
  int size;              /* number of processes in the communicator */
  int tag = 0;           /* scope for adding extra information to a message */
  MPI_Status status;     /* struct used by MPI_Recv */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */
  int remote_ncols;      /* number of columns apportioned to a remote rank */
  double *w;             /* local temperature grid at time t     */
  double *sendbuf;       /* buffer to hold values to send */
  double *recvbuf;       /* buffer to hold received values */
  double *printbuf;      /* buffer to hold values for printing */


  /* initialise our MPI environment */
  MPI_Init( &argc, &argv );

  /* check whether the initialisation was successful */
  MPI_Initialized(&flag);
  if ( flag != TRUE ) {
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  /* determine the hostname */
  MPI_Get_processor_name(hostname,&strlen_);

  /* 
  ** determine the SIZE of the group of processes associated with
  ** the 'communicator'.  MPI_COMM_WORLD is the default communicator
  ** consisting of all the processes in the launched MPI 'job'
  */
  MPI_Comm_size( MPI_COMM_WORLD, &size );
  
  /* determine the RANK of the current process [0:SIZE-1] */
  MPI_Comm_rank( MPI_COMM_WORLD, &myrank );

  /* 
  ** make use of these values in our print statement
  ** note that we are assuming that all processes can
  ** write to the screen
  */
  printf("Hello, world; from host %s: process %d of %d\n", hostname, myrank, size);

  /* 
  ** determine process ranks to the left and right of myrank
  ** respecting periodic boundary conditions
  */
  right = (myrank + 1) % size;
  left = (myrank == 0) ? (myrank + size - 1) : (myrank - 1);

  /* 
  ** determine local grid size
  ** each rank gets all the rows, but a subset of the number of columns
  */
  local_nrows = NROWS;
  local_ncols = calc_ncols_from_rank(rank, size);

  /*
  ** allocate space for:
  ** - the local grid with 2 extra columns added for the halos
  ** - message passing buffers
  ** - a buffer used to print local grid values
  */
  w = (double*)malloc(sizeof(double*) * local_nrows * (local_ncols + 2));
  sendbuf = (double*)malloc(sizeof(double) * local_nrows);
  recvbuf = (double*)malloc(sizeof(double) * local_nrows);
  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */ 
  remote_ncols = calc_ncols_from_rank(size-1, size); 
  printbuf = (double*)malloc(sizeof(double) * (remote_ncols + 2));

    /*
  ** initialize the local grid (w):
  ** - core cells are set to the value of the rank
  ** - halo cells are inititalised to a -ve value
  ** note the looping bounds for index jj is modified 
  ** to accomodate the extra halo columns
  */
  for(ii=0;ii<local_nrows;ii++) {
    for(jj=0; jj<local_ncols + 2; jj++) {
      if (jj > 0 && jj < (local_ncols + 1)) 
        w[ii * (local_ncols + 2) + jj] = (double)rank;                 /* core cells */
            else if (jj == 0 || jj == (local_ncols + 1))
        w[ii * (local_ncols + 2) + jj] = -1.0;                         /* halo cells */
    }
  }

  /*
  ** Master rank prints out the initialised grid.
  ** Proceeding row-by-row:
  ** - the master rank prints it's values, including the halo
  ** - space
  ** - the values for rows from the other ranks are received
  **   by the master, and printed
  ** - ranks other than the master send their row values to the master 
  */
  if(rank == MASTER) {
    printf("NROWS: %d\nNCOLS: %d\n",NROWS,NCOLS);
    printf("Initialised grid:\n");
  }
  for(ii=0; ii < local_nrows; ii++) {
    if(rank == MASTER) {
      for(jj=0; jj < local_ncols + 2; jj++) {
	      printf("%2.1f ",w[ii * (local_ncols + 2) + jj]);
      }
      printf(" ");
      for(kk=1; kk < size; kk++) { /* loop over other ranks */
        remote_ncols = calc_ncols_from_rank(kk, size);
        MPI_Recv(printbuf, remote_ncols + 2, MPI_DOUBLE, kk, tag, MPI_COMM_WORLD, &status);
        for(jj=0; jj < remote_ncols + 2; jj++) {
          printf("%2.1f ",printbuf[jj]);
        }
        printf(" ");
      }
      printf("\n");

    } else {
      MPI_Send(&w[ii * (local_ncols + 2)], local_ncols + 2, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
    }
  }
  if (rank == MASTER)
    printf("\n");


   /*
  ** halo exchange for the local grids w:
  ** - first send to the left and receive from the right,
  ** - then send to the right and receive from the left.
  ** for each direction:
  ** - first, pack the send buffer using values from the grid
  ** - exchange using MPI_Sendrecv()
  ** - unpack values from the recieve buffer into the grid
  */

  /* send to the left, receive from right */
  for(ii=0; ii < local_nrows; ii++)
    sendbuf[ii] = w[ii * (local_ncols + 2) + 1];
    MPI_Sendrecv(sendbuf, local_nrows, MPI_DOUBLE, left, tag,
	       recvbuf, local_nrows, MPI_DOUBLE, right, tag,
	       MPI_COMM_WORLD, &status);
  for(ii=0; ii < local_nrows; ii++)
    w[ii * (local_ncols + 2) + local_ncols + 1] = recvbuf[ii];
  
  /* send to the right, receive from left */
  for(ii=0; ii < local_nrows; ii++)
    sendbuf[ii] = w[ii * (local_ncols + 2) + local_ncols];
    MPI_Sendrecv(sendbuf, local_nrows, MPI_DOUBLE, right, tag, 
        recvbuf, local_nrows, MPI_DOUBLE, left, tag,
	       MPI_COMM_WORLD, &status);
  for(ii=0; ii < local_nrows; ii++)
    w[ii * (local_ncols + 2)] = recvbuf[ii];

  
  /*
  ** Master rank prints out the grid after the halo-exchange
  */
  if(rank == MASTER) {
    printf("Grid after halo-exchange.\n");
    printf("Notice that:\n");
    printf(" - the core cells are unchanged\n");
    printf(" - the halo cells now have values derived\n");
    printf("   from the core cells of neighbouring ranks\n");
  }
  for(ii=0; ii < local_nrows; ii++) {
    if(rank == MASTER) {
      for(jj=0; jj < local_ncols + 2; jj++) {
	printf("%2.1f ",w[ii * (local_ncols + 2) + jj]);
      }
      printf(" ");
      for(kk=1; kk < size; kk++) { /* loop over other ranks */
	remote_ncols = calc_ncols_from_rank(kk, size);
	MPI_Recv(printbuf, remote_ncols + 2, MPI_DOUBLE, kk, tag, MPI_COMM_WORLD, &status);
	for(jj=0; jj < remote_ncols + 2; jj++) {
	  printf("%2.1f ",printbuf[jj]);
	}
	printf(" ");
      }
      printf("\n");
    }
    else {
      MPI_Send(&w[ii * (local_ncols + 2)], local_ncols + 2, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
    }
  }
  if (rank == MASTER)
    printf("\n");

  char *paramfile = NULL;                                                            /* name of the input parameter file */
  char *obstaclefile = NULL;                                                         /* name of a the input obstacle file */
  t_param params;                                                                    /* struct to hold parameter values */
  t_speed* cells = NULL;                                                             /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;                                                         /* scratch space */
  unsigned int* obstacles = NULL;                                                    /* grid indicating which cells are blocked */
  float *av_vels = NULL;                                                             /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic = tot_tic;
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic = init_toc;

  accelerate_flow(params, cells, obstacles);
  is_power_of_2 = check_power_of_2(params.nx);

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    av_vels[tt] = timestep(params, cells, tmp_cells, obstacles);
    t_speed* temp = cells;
    cells = tmp_cells;
    tmp_cells = temp;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic = comp_toc;

  // Collate data from ranks here

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc - tot_tic);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  /* finialise the MPI enviroment */
  MPI_Finalize();

  /* free up allocated memory */
  free(w);
  free(sendbuf);
  free(recvbuf);
  free(printbuf);

  return EXIT_SUCCESS;
}

int calc_ncols_from_rank(int rank, int size)
{
  int ncols;

  ncols = NCOLS / size;       /* integer division */
  if ((NCOLS % size) != 0) {  /* if there is a remainder */
    if (rank == size - 1)
      ncols += NCOLS % size;  /* add remainder to last rank */
  }
  
  return ncols;
}

int accelerate_flow(const t_param params, t_speed* restrict cells, unsigned int* restrict obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj * params.nx] 
    && (cells->speed_3[ii + jj * params.nx] - w1) > 0.f 
    && (cells->speed_6[ii + jj * params.nx] - w2) > 0.f 
    && (cells->speed_7[ii + jj * params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speed_1[ii + jj * params.nx] += w1;
      cells->speed_5[ii + jj * params.nx] += w2;
      cells->speed_8[ii + jj * params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speed_3[ii + jj * params.nx] -= w1;
      cells->speed_6[ii + jj * params.nx] -= w2;
      cells->speed_7[ii + jj * params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, unsigned int* restrict obstacles)
{
  unsigned int tot_cells = 0; /* no. of cells used in calculation */
  float tot_u = 0.f;       /* accumulated magnitudes of velocity for each cell */
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  /* compute weighting factors */
  float w1_ = params.density * params.accel / 9.f;
  float w2_ = params.density * params.accel / 36.f;

  __assume_aligned(obstacles, 64);

  __assume_aligned(cells->speed_0, 64);
  __assume_aligned(cells->speed_1, 64);
  __assume_aligned(cells->speed_2, 64);
  __assume_aligned(cells->speed_3, 64);
  __assume_aligned(cells->speed_4, 64);
  __assume_aligned(cells->speed_5, 64);
  __assume_aligned(cells->speed_6, 64);
  __assume_aligned(cells->speed_7, 64);
  __assume_aligned(cells->speed_8, 64);

  __assume_aligned(tmp_cells->speed_0, 64);
  __assume_aligned(tmp_cells->speed_1, 64);
  __assume_aligned(tmp_cells->speed_2, 64);
  __assume_aligned(tmp_cells->speed_3, 64);
  __assume_aligned(tmp_cells->speed_4, 64);
  __assume_aligned(tmp_cells->speed_5, 64);
  __assume_aligned(tmp_cells->speed_6, 64);
  __assume_aligned(tmp_cells->speed_7, 64);
  __assume_aligned(tmp_cells->speed_8, 64);
  __assume((params.nx)%2==0);
  __assume((params.ny)%2==0);

  #pragma omp simd reduction(+:tot_cells, tot_u)
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      __assume((obstacles[jj*params.nx + ii])<2);
      unsigned int y_n = (jj+1 == params.ny) ? 0 : (jj+1);
      unsigned int x_e = (ii+1 == params.nx) ? 0 : (ii+1);
      unsigned int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
      unsigned int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
      __assume(y_n < 128);
      __assume(x_e < 128);
      __assume(y_s < params.ny - 1);
      __assume(x_w < params.nx - 1);

      register float speed_0 = cells->speed_0[ii + jj*params.nx];   /* central cell, no movement */
      register float speed_1 = cells->speed_1[x_w + jj*params.nx];  /* east */
      register float speed_2 = cells->speed_2[ii + y_s*params.nx];  /* north */
      register float speed_3 = cells->speed_3[x_e + jj*params.nx];  /* west */
      register float speed_4 = cells->speed_4[ii + y_n*params.nx];  /* south */
      register float speed_5 = cells->speed_5[x_w + y_s*params.nx]; /* north-east */
      register float speed_6 = cells->speed_6[x_e + y_s*params.nx]; /* north-west */
      register float speed_7 = cells->speed_7[x_e + y_n*params.nx]; /* south-west */
      register float speed_8 = cells->speed_8[x_w + y_n*params.nx]; /* south-east */

      /**If cell contains an obstacle rebound else collision occurs */
      if (obstacles[jj*params.nx + ii])
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */

        tmp_cells->speed_1[ii + jj*params.nx] = speed_3;
        tmp_cells->speed_2[ii + jj*params.nx] = speed_4;
        tmp_cells->speed_3[ii + jj*params.nx] = speed_1;
        tmp_cells->speed_4[ii + jj*params.nx] = speed_2;
        tmp_cells->speed_5[ii + jj*params.nx] = speed_7;
        tmp_cells->speed_6[ii + jj*params.nx] = speed_8;
        tmp_cells->speed_7[ii + jj*params.nx] = speed_5;
        tmp_cells->speed_8[ii + jj*params.nx] = speed_6;

      } else {
        register float local_density = speed_0 + speed_1 + speed_2 + speed_3 + speed_4 + speed_5 + speed_6 + speed_7 + speed_8;

        /* compute x velocity component */
        register float u_x = (speed_1 + speed_5 + speed_8 - (speed_3 + speed_6 + speed_7)) / local_density;
        /* compute y velocity component */
        register float u_y = (speed_2 + speed_5 + speed_6 - (speed_4 + speed_7 + speed_8)) / local_density;
        
        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* zero velocity density: weight w0 */
        register float d_equ = w0 * local_density
                    * (1.f - u_sq / (2.f * c_sq));
        speed_0 = speed_0 + params.omega * (d_equ - speed_0);   

        /* axis speeds: weight w1 */
        d_equ = w1 * local_density * (1.f + u_x / c_sq
                                          + (u_x * u_x) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        speed_1 = speed_1 + params.omega * (d_equ - speed_1);

        d_equ = w1 * local_density * (1.f + u_y / c_sq
                                          + (u_y * u_y) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        speed_2 = speed_2 + params.omega * (d_equ - speed_2);

        d_equ = w1 * local_density * (1.f + -u_x / c_sq
                                          + (-u_x * -u_x) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        speed_3 = speed_3 + params.omega * (d_equ - speed_3);

        d_equ = w1 * local_density * (1.f + -u_y / c_sq
                                          + (-u_y * -u_y) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        speed_4 = speed_4 + params.omega * (d_equ - speed_4);                                  
        /* diagonal speeds: weight w2 */
        d_equ = w2 * local_density * (1.f + (u_x + u_y) / c_sq
                                          + ((u_x + u_y) * (u_x + u_y)) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
          
        speed_5 = speed_5 + params.omega * (d_equ - speed_5);

        d_equ = w2 * local_density * (1.f + (-u_x + u_y) / c_sq
                                          + ((-u_x + u_y) * (-u_x + u_y)) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        speed_6 = speed_6 + params.omega * (d_equ - speed_6);

        d_equ = w2 * local_density * (1.f + (-u_x - u_y) / c_sq
                                          + ((-u_x - u_y) * (-u_x - u_y)) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));
        speed_7 = speed_7 + params.omega * (d_equ - speed_7);

        d_equ = w2 * local_density * (1.f + (u_x - u_y) / c_sq
                                          + ((u_x - u_y) * (u_x - u_y)) / (2.f * c_sq * c_sq)
                                          - u_sq / (2.f * c_sq));         
        speed_8 = speed_8 + params.omega * (d_equ - speed_8);                                  
        
        local_density = speed_0 + speed_1 + speed_2 + speed_3 + speed_4 + speed_5 + speed_6 + speed_7 + speed_8;

        /* compute x velocity component */
        u_x = (speed_1 + speed_5 + speed_8 - (speed_3 + speed_6 + speed_7)) / local_density;
        /* compute y velocity component */
        u_y = (speed_2 + speed_5 + speed_6 - (speed_4 + speed_7 + speed_8)) / local_density;

        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;


        /* if the cell is not occupied and
        ** we don't send a negative density */
        if (jj == params.ny-2 
        && !obstacles[jj*params.nx + ii]
        && (speed_3 - w1_) > 0.f 
        && (speed_6 - w2_) > 0.f 
        && (speed_7 - w2_) > 0.f)
        {
          /* increase 'east-side' densities */
          speed_1 += w1_;
          speed_5 += w2_;
          speed_8 += w2_;
          /* decrease 'west-side' densities */
          speed_3 -= w1_;
          speed_6 -= w2_;
          speed_7 -= w2_;
        }

        /* write back new state from speed variables to tmp_cells ready for next iteration */
        tmp_cells->speed_0[ii + jj*params.nx] = speed_0;
        tmp_cells->speed_1[ii + jj*params.nx] = speed_1;
        tmp_cells->speed_2[ii + jj*params.nx] = speed_2;
        tmp_cells->speed_3[ii + jj*params.nx] = speed_3;
        tmp_cells->speed_4[ii + jj*params.nx] = speed_4;
        tmp_cells->speed_5[ii + jj*params.nx] = speed_5;
        tmp_cells->speed_6[ii + jj*params.nx] = speed_6;
        tmp_cells->speed_7[ii + jj*params.nx] = speed_7;
        tmp_cells->speed_8[ii + jj*params.nx] = speed_8;

      }
    }
  }

  return tot_u / (float)tot_cells;
}


float av_velocity(const t_param params, t_speed* restrict cells, unsigned int* restrict obstacles)
{
  unsigned int tot_cells = 0; /* no. of cells used in calculation */
  float tot_u;       /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (unsigned int jj = 0; jj < params.ny; jj++)
  {
    for (unsigned int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj * params.nx])
      {
        float speed_0 = cells->speed_0[ii + jj*params.nx];
        float speed_1 = cells->speed_1[ii + jj*params.nx];
        float speed_2 = cells->speed_2[ii + jj*params.nx];
        float speed_3 = cells->speed_3[ii + jj*params.nx];
        float speed_4 = cells->speed_4[ii + jj*params.nx];
        float speed_5 = cells->speed_5[ii + jj*params.nx];
        float speed_6 = cells->speed_6[ii + jj*params.nx];
        float speed_7 = cells->speed_7[ii + jj*params.nx];
        float speed_8 = cells->speed_8[ii + jj*params.nx];
        /* local density total */
        float local_density = speed_0 + speed_1 + speed_2 + speed_3 + speed_4 + speed_5 + speed_6 + speed_7 + speed_8;

         /* compute x velocity component */
        float u_x = (speed_1 + speed_5 + speed_8 - (speed_3 + speed_6 + speed_7)) / local_density;
        /* compute y velocity component */
        float u_y = (speed_2 + speed_5 + speed_6 - (speed_4 + speed_7 + speed_8)) / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}


int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
               unsigned int **obstacles_ptr, float **av_vels_ptr)
{
  char message[1024];   /* message buffer */
  FILE *fp;             /* file pointer */
  int xx, yy;           /* generic array indices */
  unsigned int blocked; /* indicates whether a cell is blocked by an obstacle */
  int retval;           /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1)
    die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1)
    die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1)
    die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1)
    die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1)
    die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1)
    die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1)
    die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed *)malloc(sizeof(t_speed));
  (*cells_ptr)->speed_0 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_1 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_2 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_3 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_4 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_5 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_6 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_7 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*cells_ptr)->speed_8 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*cells_ptr == NULL)
    die("cannot allocate memory for cells", __LINE__, __FILE__);

  if ((*cells_ptr)->speed_0 == NULL 
  || (*cells_ptr)->speed_1 == NULL 
  || (*cells_ptr)->speed_2 == NULL 
  || (*cells_ptr)->speed_3 == NULL 
  || (*cells_ptr)->speed_4 == NULL
  || (*cells_ptr)->speed_5 == NULL 
  || (*cells_ptr)->speed_6 == NULL 
  || (*cells_ptr)->speed_7 == NULL 
  || (*cells_ptr)->speed_8 == NULL)
    die("cannot allocate memory for a speed in cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed *)malloc(sizeof(t_speed));
  (*tmp_cells_ptr)->speed_0 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_1 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_2 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_3 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_4 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_5 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_6 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_7 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  (*tmp_cells_ptr)->speed_8 = (float *)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*tmp_cells_ptr == NULL)
    die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  if ((*tmp_cells_ptr)->speed_0 == NULL 
  || (*tmp_cells_ptr)->speed_1 == NULL 
  || (*tmp_cells_ptr)->speed_2 == NULL 
  || (*tmp_cells_ptr)->speed_3 == NULL 
  || (*tmp_cells_ptr)->speed_4 == NULL
  || (*tmp_cells_ptr)->speed_5 == NULL 
  || (*tmp_cells_ptr)->speed_6 == NULL 
  || (*tmp_cells_ptr)->speed_7 == NULL 
  || (*tmp_cells_ptr)->speed_8 == NULL)
    die("cannot allocate memory for a speed in tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = (unsigned int *)_mm_malloc(sizeof(unsigned int) * (params->ny * params->nx), 64);

  if (*obstacles_ptr == NULL)
    die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density / 9.f;
  float w2 = params->density / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
       /* centre */
      (*cells_ptr)->speed_0[ii + jj * params->nx] = w0;
      /* axis directions */
      (*cells_ptr)->speed_1[ii + jj * params->nx] = w1;
      (*cells_ptr)->speed_2[ii + jj * params->nx] = w1;
      (*cells_ptr)->speed_3[ii + jj * params->nx] = w1;
      (*cells_ptr)->speed_4[ii + jj * params->nx] = w1;
      /* diagonals */
      (*cells_ptr)->speed_5[ii + jj * params->nx] = w2;
      (*cells_ptr)->speed_6[ii + jj * params->nx] = w2;
      (*cells_ptr)->speed_7[ii + jj * params->nx] = w2;
      (*cells_ptr)->speed_8[ii + jj * params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj * params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3)
      die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1)
      die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1)
      die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1)
      die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy * params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float *)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
             unsigned int **obstacles_ptr, float **av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed* restrict cells, unsigned int* restrict obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* restrict cells)
{
  float total = 0.f; /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total += cells->speed_0[ii + jj * params.nx] 
      + cells->speed_1[ii + jj * params.nx] + cells->speed_2[ii + jj * params.nx] 
      + cells->speed_3[ii + jj * params.nx] + cells->speed_4[ii + jj * params.nx] 
      + cells->speed_5[ii + jj * params.nx] + cells->speed_6[ii + jj * params.nx] 
      + cells->speed_7[ii + jj * params.nx] + cells->speed_8[ii + jj * params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* restrict cells, unsigned int* restrict obstacles, float* restrict av_vels)
{
  FILE *fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float pressure;               /* fluid pressure in grid cell */
  float u_x;                    /* x-component of velocity in grid cell */
  float u_y;                    /* y-component of velocity in grid cell */
  float u;                      /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj * params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        float speed_0 = cells->speed_0[ii + jj*params.nx];
        float speed_1 = cells->speed_1[ii + jj*params.nx];
        float speed_2 = cells->speed_2[ii + jj*params.nx];
        float speed_3 = cells->speed_3[ii + jj*params.nx];
        float speed_4 = cells->speed_4[ii + jj*params.nx];
        float speed_5 = cells->speed_5[ii + jj*params.nx];
        float speed_6 = cells->speed_6[ii + jj*params.nx];
        float speed_7 = cells->speed_7[ii + jj*params.nx];
        float speed_8 = cells->speed_8[ii + jj*params.nx];

        /* local density total */
        float local_density = speed_0 + speed_1 + speed_2 + speed_3 + speed_4 + speed_5 + speed_6 + speed_7 + speed_8;

         /* compute x velocity component */
        float u_x = (speed_1 + speed_5 + speed_8 - (speed_3 + speed_6 + speed_7)) / local_density;
        /* compute y velocity component */
        float u_y = (speed_2 + speed_5 + speed_6 - (speed_4 + speed_7 + speed_8)) / local_density;
       
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char *message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char *exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
