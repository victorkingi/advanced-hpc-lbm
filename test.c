#include <stdio.h>

double calculate_pi(int num_steps) {
    double step, sum, pi;
    sum = 0;
    step = 1.0 / num_steps;

    #pragma omp parallel for reduction(+:sum)
    for (int ii = 1; ii <= num_steps; ++ii) {
        double x = (ii - 0.5) * step;
        sum = sum + (4.0/(1.0+x*x));
    }
    pi = step * sum;
    return pi;
}

int main(int argc, char* argv[]) {
    double pi = calculate_pi(100000000);
    printf("%lf\n", pi);
    return 0;
}