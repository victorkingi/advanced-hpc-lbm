#include <stdio.h>
#include <math.h>

int main() {
    int ans = 89 ^ 4;
    int shift = 1-4;
    printf("expected: %d\n", ans);
    printf("found: %d\n", shift);

    return 0;
}