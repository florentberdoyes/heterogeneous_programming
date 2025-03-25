#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Cuda : 
#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 16

// Gaussian function
double gaussian(double x, double sigma) {
    return exp(-(x * x) / (2.0 * sigma * sigma));
}

// Manual bilateral filter
void bilateral_filter(unsigned char *src, unsigned char *dst, int width, int height, int channels, int d, double sigma_color, double sigma_space) {
    int radius = d / 2;

    // Precompute spatial Gaussian weights
    double *spatial_weights = (double *)malloc(d * d * sizeof(double));
    if (!spatial_weights) {
        printf("Memory allocation for spatial weights failed!\n");
        return;
    }

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int x = i - radius, y = j - radius;
            spatial_weights[i * d + j] = gaussian(sqrt(x * x + y * y), sigma_space);
        }
    }

    int image_size = width * height * channels * sizeof(unsigned char);
    int weight_size = (2 * radius + 1) * (2 * radius + 1) * sizeof(double);
    
    // Allocate memory on GPU
    unsigned char *d_src, *d_dst;
    double *d_spatial_weights;

    cudaMalloc((void **)&d_src, image_size);
    cudaMalloc((void **)&d_dst, image_size);
    cudaMalloc((void **)&d_spatial_weights, weight_size);

    // Copy data to GPU
    cudaMemcpy(d_src, h_src, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatial_weights, h_spatial_weights, weight_size, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the CUDA kernel
    bilateral_filter_kernel<<<grid, block>>>(d_src, d_dst, width, height, channels, radius, 2 * radius + 1, sigma_color, d_spatial_weights);

    // Copy the result back to CPU
    cudaMemcpy(h_dst, d_dst, image_size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_spatial_weights);
    

    free(spatial_weights);
}

// Main function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    clock_t start_time = clock();

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, 0);
    if (!image) {
        printf("Error loading image!\n");
        return 1;
    }

    // Ensure that image is not too small for bilateral filter (at least radius of d/2 around edges)
    if (width <= 5 || height <= 5) {
        printf("Image is too small for bilateral filter (at least 5x5 size needed).\n");
        stbi_image_free(image);
        return 1;
    }

    // Allocate memory for output image
    unsigned char *filtered_image = (unsigned char *)malloc(width * height * channels);
    if (!filtered_image) {
        printf("Memory allocation for filtered image failed!\n");
        stbi_image_free(image);
        return 1;
    }
    
    // Apply the bilateral filter
    bilateral_filter(image, filtered_image, width, height, channels, 5, 75.0, 75.0);

    // Save the output image
    if (!stbi_write_png(argv[2], width, height, channels, filtered_image, width * channels)) {
        printf("Error saving the image!\n");
        free(filtered_image);
        stbi_image_free(image);
        return 1;
    }

    // Free memory
    stbi_image_free(image);
    free(filtered_image);

    clock_t end_time = clock(); // Fin du timer
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Bilateral filtering complete. Output saved as %s\n", argv[2]);
    printf("Execution time: %.4f seconds\n", elapsed_time);
    
    return 0;
}

__global__ void bilateral_filter_kernel(
    unsigned char *src, unsigned char *dst, int width, int height, int channels,
    int radius, int d, double sigma_color, double *spatial_weights) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= radius && x < width - radius && y >= radius && y < height - radius) {
        double weight_sum[3] = {0.0, 0.0, 0.0};
        double filtered_value[3] = {0.0, 0.0, 0.0};

        unsigned char *center_pixel = src + (y * width + x) * channels;

        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                int nx = x + j - radius;
                int ny = y + i - radius;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    unsigned char *neighbor_pixel = src + (ny * width + nx) * channels;

                    for (int c = 0; c < channels; c++) {
                        double range_weight = exp(-(pow((double)(neighbor_pixel[c] - center_pixel[c]), 2)) / (2.0 * sigma_color * sigma_color));
                        double weight = spatial_weights[i * d + j] * range_weight;

                        filtered_value[c] += neighbor_pixel[c] * weight;
                        weight_sum[c] += weight;
                    }
                }
            }
        }

        unsigned char *output_pixel = dst + (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
            output_pixel[c] = (unsigned char)(filtered_value[c] / (weight_sum[c] + 1e-6));
        }
    }
}

