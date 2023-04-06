#include <iostream>
#include "Vector.h"
#include "Matrix.h"

int main() {
    // Test default constructor
    Vector<int> vec1;
    std::cout << "vec1 (default constructor): " << vec1 << std::endl;

    // Test constructor with size and default value
    Vector<int> vec2(5, 7);
    std::cout << "vec2 (constructor with size and default value): " << vec2 << std::endl;

    // Test constructor with initializer list
    Vector<int> vec3({1, 2, 3, 4, 5});
    std::cout << "vec3 (constructor with initializer list): " << vec3 << std::endl;

    // Test copy constructor
    Vector<int> vec4(vec3);
    std::cout << "vec4 (copy constructor): " << vec4 << std::endl;

    // Test move constructor
    Vector<int> vec5(std::move(vec4));
    std::cout << "vec5 (move constructor): " << vec5 << std::endl;
    std::cout << "vec4 (after move constructor): " << vec4 << std::endl;

    // Test assignment operators
    vec1 = vec2;
    std::cout << "vec1 (copy assignment operator): " << vec1 << std::endl;

    vec2 = std::move(vec1);
    std::cout << "vec2 (move assignment operator): " << vec2 << std::endl;
    std::cout << "vec1 (after move assignment operator): " << vec1 << std::endl;

    // Test arithmetic operators
    std::cout << "vec3 + vec5: " << vec3 + vec5 << std::endl;
    std::cout << "vec3 - vec5: " << vec3 - vec5 << std::endl;
    std::cout << "vec3 * 2: " << vec3 * 2 << std::endl;
    std::cout << "2 * vec3: " << 2 * vec3 << std::endl;
    std::cout << "vec3 / 2: " << vec3 / 2 << std::endl;
    std::cout << "vec3 * vec5 (dot product): " << (vec3 * vec5) << std::endl;

    // Test compound assignment operators
    vec3 += vec5;
    std::cout << "vec3 (after vec3 += vec5): " << vec3 << std::endl;

    vec3 -= vec5;
    std::cout << "vec3 (after vec3 -= vec5): " << vec3 << std::endl;

    vec3 *= 2;
    std::cout << "vec3 (after vec3 *= 2): " << vec3 << std::endl;

    vec3 /= 2;
    std::cout << "vec3 (after vec3 /= 2): " << vec3 << std::endl;

    // Test equality and inequality operators
    std::cout << "vec3 == vec5: " << (vec3 == vec5) << std::endl;
    std::cout << "vec3 != vec5: " << (vec3 != vec5) << std::endl;

    // Test default constructor
    Matrix<int> mat1;
    std::cout << "mat1 (default constructor):\n" << mat1 << std::endl;

    // Test constructor with size and default value
    Matrix<int> mat2(3, 3, 7);
    std::cout << "mat2 (constructor with size and default value):\n" << mat2 << std::endl;

    // Test constructor with initializer list
    Matrix<int> mat3({ {1, 2, 3},
                       {4, 5, 6},
                       {7, 8, 9} });
    std::cout << "mat3 (constructor with initializer list):\n" << mat3 << std::endl;

    // Test copy constructor
    Matrix<int> mat4(mat3);
    std::cout << "mat4 (copy constructor):\n" << mat4 << std::endl;

    // Test move constructor
    Matrix<int> mat5(std::move(mat4));
    std::cout << "mat5 (move constructor):\n" << mat5 << std::endl;
    std::cout << "mat4 (after move constructor):\n" << mat4 << std::endl;

    // Test assignment operators
    mat1 = mat2;
    std::cout << "mat1 (copy assignment operator):\n" << mat1 << std::endl;

    mat2 = std::move(mat1);
    std::cout << "mat2 (move assignment operator):\n" << mat2 << std::endl;
    std::cout << "mat1 (after move assignment operator):\n" << mat1 << std::endl;

    // Test arithmetic operators
    std::cout << "mat3 + mat5:\n" << mat3 + mat5 << std::endl;
    std::cout << "mat3 - mat5:\n" << mat3 - mat5 << std::endl;
    std::cout << "mat3 * 2:\n" << mat3 * 2 << std::endl;
    std::cout << "2 * mat3:\n" << 2 * mat3 << std::endl;
    std::cout << "mat3 / 2:\n" << mat3 / 2 << std::endl;
    std::cout << "mat3 * mat5 (matrix multiplication):\n" << mat3 * mat5 << std::endl;

    // Test matrix-vector multiplication
    Vector<int> vec({1, 2, 3});
    std::cout << "mat3 * vec:\n" << mat3 * vec << std::endl;

    // Test compound assignment operators
    mat3 += mat5;
    std::cout << "mat3 (after mat3 += mat5):\n" << mat3 << std::endl;

    mat3 -= mat5;
    std::cout << "mat3 (after mat3 -= mat5):\n" << mat3 << std::endl;

    mat3 *= 2;
    std::cout << "mat3 (after mat3 *= 2):\n" << mat3 << std::endl;

    mat3 /= 2;
    std::cout << "mat3 (after mat3 /= 2):\n" << mat3 << std::endl;

    // Test equality and inequality operators
    std::cout << "mat3 == mat5: " << (mat3 == mat5) << std::endl;
    std::cout << "mat3 != mat5: "<< (mat3 != mat5) << std::endl;

    Matrix<double> mat6({ {4, 7},
                          {2, 6} });
    std::cout << "Inverse of mat6:\n" << inverse(mat6) << std::endl;
    std::cout << "Determinant of mat6: " << determinant(mat6) << std::endl;

    return 0;
}