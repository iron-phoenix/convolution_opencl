__kernel void convolution(__global float * A, __global float * B, __global float * C, int a_size, int b_size) {
   int row = get_global_id(0);
   int col = get_global_id(1);

   if (row >= a_size || col >= a_size)
        return;

   float result = 0.0;
   for (int i = 0; i < b_size; ++i) {
	   for (int j = 0; j < b_size; ++j) {
		   int x = row + i - b_size / 2;
		   int y = col + j - b_size / 2;
		   if (x >= 0 && x < a_size && y >= 0 && y < a_size)
			   result += A[x * a_size + y] * B[i * b_size + j];
	   }
   }

   C[row * a_size + col] = result;
}