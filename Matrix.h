//
// Created by ustym on 06-Apr-23.
//
#include <immintrin.h> // for SIMD intrinsics
#include <omp.h> // for OpenMP pragmas
#ifndef UNTITLED3_MATRIX_H
// Forward declaration
template<typename T>
class Matrix;

// Scalar multiplication and division operators
template<typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& mat);

template<typename T>
Matrix<T> operator*(const Matrix<T>& mat, const T& scalar);

template<typename T>
Matrix<T> operator/(const Matrix<T>& mat, const T& scalar);

// A class template for matrices of arbitrary size and type
template<typename T>
class Matrix {
private:
    // The number of rows and columns of the matrix
    size_t rows_, cols_;
    // The data of the matrix stored in a dynamic array of vectors
    Vector<T> *data_;
public:
    // Default constructor that creates an empty matrix
    Matrix();

    // Constructor that creates a matrix of a given size and fills it with a default value
    Matrix(size_t rows, size_t cols, const T &value = T());

    // Constructor that creates a matrix from an initializer list of vectors
    Matrix(std::initializer_list<Vector<T>> list);

    // Copy constructor that creates a deep copy of another matrix
    Matrix(const Matrix<T> &other);

    // Move constructor that transfers ownership of another matrix's data
    Matrix(Matrix<T> &&other);

    // Destructor that frees the allocated memory
    ~Matrix();


    // Copy assignment operator that creates a deep copy of another matrix
    Matrix<T> &operator=(const Matrix<T> &other);

    // Move assignment operator that transfers ownership of another matrix's data
    Matrix<T> &operator=(Matrix<T> &&other);

    // Subscript operator that returns a reference to a vector at a given row index
    Vector<T> &operator[](size_t index);

    // Subscript operator that returns a const reference to a vector at a given row index
    const Vector<T> &operator[](size_t index) const;

    // Unary plus operator that returns a copy of the matrix
    Matrix<T> operator+() const;

    // Unary minus operator that returns the negation of the matrix
    Matrix<T> operator-() const;

    // Binary plus operator that returns the sum of two matrices
    Matrix<T> operator+(const Matrix<T> &other) const;

    // Binary minus operator that returns the difference of two matrices
    Matrix<T> operator-(const Matrix<T> &other) const;

    // Scalar multiplication operator that returns the product of a matrix and a scalar
    template<typename U>
    friend Matrix<U> operator*(const U& scalar, const Matrix<U>& mat);

    template<typename U>
    friend Matrix<U> operator*(const Matrix<U>& mat, const U& scalar);

    // Scalar division operator that returns the quotient of a matrix and a scalar
    template<typename U>
    friend Matrix<U> operator/(const Matrix<U>& mat, const U& scalar);

    // Matrix multiplication operator that returns the product of two matrices
    Matrix<T> operator*(const Matrix<T> &other) const;

    // Vector multiplication operator that returns the product of a matrix and a vector
    Vector<T> operator*(const Vector<T> &vec) const;

    // Compound assignment operators that modify the matrix in place
    Matrix<T> &operator+=(const Matrix<T> &other);

    Matrix<T> &operator-=(const Matrix<T> &other);

    template<typename U>
    Matrix<T> &operator*=(const U &scalar);

    template<typename U>
    Matrix<T> &operator/=(const U &scalar);

    // Equality operator that returns true if two matrices have the same size and elements
     bool operator==(const Matrix<T> &other) const;

    // Inequality operator that returns true if two matrices are not equal
     bool operator!=(const Matrix<T> &other) const;

    // Stream insertion operator that prints the matrix to an output stream
    template<typename U>
    friend std::ostream &operator<<(std::ostream &os, const Matrix<U> &mat);

    // Stream extraction operator that reads the matrix from an input stream
    template<typename U>
    friend std::istream &operator>>(std::istream &is, Matrix<U> &mat);

    // A function that returns the number of rows of the matrix
    size_t rows() const;

    // A function that returns the number of columns of the matrix
    size_t cols() const;

    // A function that returns a pointer to the data of the matrix
    Vector<T> *data();

    // A function that returns a const pointer to the data of the matrix
    const Vector<T> *data() const;

    T& operator()(size_t row, size_t col) {
        return data_[row][col];
    }

    const T& operator()(size_t row, size_t col) const {
        return data_[row][col];
    }

    static Matrix identity(size_t n) {
        Matrix I(n, n);
        for (size_t i = 0; i < n; ++i) {
            I(i, i) = 1.0f;
        }
        return I;
    }
    void swap_rows(size_t i, size_t j);
    void scale_row(size_t i, T factor);
    void add_row(size_t i, size_t j, T factor);
};

template<typename T>
// A function that returns the number of rows of the matrix
size_t Matrix<T>::rows() const {
    return rows_;
}
template<typename T>
// A function that returns the number of columns of the matrix
size_t Matrix<T>::cols() const {
    return cols_;
}
template<typename T>
// A function that returns a pointer to the data of the matrix
Vector<T> Matrix<T>::*data() {
    return data;
}
template<typename T>
// A function that returns a const pointer to the data of the matrix
const Vector<T> Matrix<T>::*data() {
    return data;
}

// Default constructor
template<typename T>
Matrix<T>::Matrix() : rows_(0), cols_(0), data_(nullptr) {}

// Constructor with given size
template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T& value)
        : rows_(rows), cols_(cols), data_(new Vector<T>[rows]) {
    for (size_t i = 0; i < rows_; ++i) {
        data_[i] = Vector<T>(cols, value);
    }
}

// Constructor from an initializer list
template<typename T>
Matrix<T>::Matrix(std::initializer_list<Vector<T>> list)
        : rows_(list.size()), cols_(list.begin()->size()), data_(new Vector<T>[rows_]) {
    std::copy(list.begin(), list.end(), data_);
}

// Copy constructor
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other)
        : rows_(other.rows_), cols_(other.cols_), data_(new Vector<T>[rows_]) {
    for (size_t i = 0; i < rows_; ++i) {
        data_[i] = other.data_[i];
    }
}

// Move constructor
template<typename T>
Matrix<T>::Matrix(Matrix<T>&& other)
        : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
    other.rows_ = 0;
    other.cols_ = 0;
    other.data_ = nullptr;
}

// Destructor
template<typename T>
Matrix<T>::~Matrix() {
    delete[] data_;
}

// Copy assignment operator
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
    if (this != &other) {
        if (rows_ != other.rows_ || cols_ != other.cols_) {
            delete[] data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = new Vector<T>[rows_];
        }
        for (size_t i = 0; i < rows_; ++i) {
            data_[i] = other.data_[i];
        }
    }
    return *this;
}

// Move assignment operator
template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix<T>&& other) {
    if (this != &other) {
        delete[] data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

// Subscript operator
template<typename T>
Vector<T>& Matrix<T>::operator[](size_t index) {
    if (index >= rows_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[index];
}

template<typename T>
const Vector<T>& Matrix<T>::operator[](size_t index) const {
    if (index >= rows_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[index];
}

// Unary plus and minus operators
template<typename T>
Matrix<T> Matrix<T>::operator+() const {
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator-() const {
    Matrix<T> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i) {
        result.data_[i] = -data_[i];
    }
    return result;
}

// Binary plus and minus operators
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }
    Matrix<T> result(rows_, cols_);
#pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction");
    }
    Matrix<T> result(rows_, cols_);
#pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

// Scalar multiplication and division operators
template<typename T>
Matrix<T> operator*(const T& scalar, const Matrix<T>& mat) {
    Matrix<T> result(mat.rows_, mat.cols_);
#pragma omp parallel for
    for (size_t i = 0; i < mat.rows_; ++i) {
        result.data_[i] = scalar * mat.data_[i];
    }
    return result;
}

template<typename T>
Matrix<T> operator*(const Matrix<T>& mat, const T& scalar) {
    return scalar * mat;
}

template<typename T>
Matrix<T> operator/(const Matrix<T>& mat, const T& scalar) {
    Matrix<T> result(mat.rows_, mat.cols_);
#pragma omp parallel for
    for (size_t i = 0; i < mat.rows_; ++i) {
        result.data_[i] = mat.data_[i] / scalar;
    }
    return result;
}

// Matrix multiplication operator
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    Matrix<T> result(rows_, other.cols_, T());
#pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            for (size_t k = 0; k < cols_; ++k) {
                result.data_[i][j] += data_[i][k] * other.data_[k][j];
            }
        }
    }
    return result;
}


// Vector multiplication operator
template<typename T>
Vector<T> Matrix<T>::operator*(const Vector<T>& vec) const {
    if (cols_ != vec.size()) {
        throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication");
    }
    Vector<T> result(rows_, T());
#pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result[i] += data_[i][j] * vec[j];
        }
    }
    return result;
}

// Compound assignment operators
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for addition");
    }
#pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other) {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction");
    }
#pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

template<typename T>
template<typename U>
Matrix<T>& Matrix<T>::operator*=(const U& scalar) {
#pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

template<typename T>
template<typename U>
Matrix<T>& Matrix<T>::operator/=(const U& scalar) {
#pragma omp parallel for
    for (size_t i = 0; i < rows_; ++i) {
        data_[i] /= scalar;
    }
    return *this;
}





template<typename T>
__m256d loadu(const T* ptr);

template<>
__m256d loadu<double>(const double* ptr) {
    return _mm256_loadu_pd(ptr);
}

template<>
__m256d loadu<float>(const float* ptr) {
    return _mm256_castps_pd(_mm256_loadu_ps(ptr));
}

template<typename T>
void storeu(T* ptr, __m256d vec);

template<>
void storeu<double>(double* ptr, __m256d vec) {
    _mm256_storeu_pd(ptr, vec);
}

template<>
void storeu<float>(float* ptr, __m256d vec) {
    _mm256_storeu_ps(ptr, _mm256_castpd_ps(vec));
}


// Equality operators
template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
    // Check if the matrices have the same dimensions, otherwise return false
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        return false;
    }

    // Get the number of rows and columns of the matrices
    int n = rows_;
    int m = cols_;

    // Initialize a flag to true
    bool equal = true;

    // Use OpenMP pragmas to parallelize the comparison process
#pragma omp parallel for reduction(&:equal)
    for (int i = 0; i < n; i++) {
        // Use SIMD intrinsics to perform vectorized operations on each row
        __m256d* lhs_row_ptr = (__m256d*) &data_[i * cols_]; // get a pointer to the current row of the left-hand side matrix
        __m256d* rhs_row_ptr = (__m256d*) &other.data_[i * cols_]; // get a pointer to the current row of the right-hand side matrix
        int vec_size = m / 4; // get the number of vectorized operations

        // Loop over each vectorized operation
        for (int k = 0; k < vec_size; k++) {
            __m256d lhs_row_vec = loadu((double*) lhs_row_ptr + k); // load a vector from the current row of the left-hand side matrix
            __m256d rhs_row_vec = loadu((double*) rhs_row_ptr + k); // load a vector from the current row of the right-hand side matrix
            __m256d cmp_vec = _mm256_cmp_pd(lhs_row_vec, rhs_row_vec, _CMP_EQ_OQ); // compare the vectors for equality and set the corresponding bits
            int cmp_mask = _mm256_movemask_pd(cmp_vec); // move the comparison bits to an integer mask
            if (cmp_mask != 0xF) { // check if all bits are set to 1
                equal = false; // update the flag to false
                break; // exit the loop early
            }
        }

        // Handle the remaining elements that are not divisible by 4
        for (int k = vec_size * 4; k < m; k++) {
            if (data_[i * cols_ + k] != other.data_[i * cols_ + k]) { // compare the elements for equality
                equal = false; // update the flag to false
                break; // exit the loop early
            }
        }
    }

    return equal;
}

// Inequality operator
template <typename T>
bool Matrix<T>::operator!=(const Matrix<T>& other) const {
    return !(*this == other); // use the negation of the equality operator
}

// I/O operators
template<typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &mat) {
    // Loop over each row of the matrix
    for (size_t i = 0; i < mat.rows(); i++) {
        // Loop over each column of the matrix
        for (size_t j = 0; j < mat.cols(); j++) {
            // Print the element at the current position with a space
            os << mat[i][j] << " ";
        }
        // Print a newline after each row
        os << "\n";
    }
    // Return the output stream
    return os;
}

// Helper functions for matrix operations

template <typename T>
void Matrix<T>::swap_rows(size_t i, size_t j) {
    if (i != j) {
        Vector<T> temp = data_[i];
        data_[i] = data_[j];
        data_[j] = temp;
    }
}

template <typename T>
void Matrix<T>::scale_row(size_t i, T factor) {
    // Use SIMD intrinsics to scale a row by a factor
    size_t k = cols_ / 4; // Number of SIMD vectors in a row
    __m128 f = _mm_set1_ps(factor); // Load the factor into a SIMD vector
    __m128 *p = (__m128 *)data_[i].data(); // Pointer to the row data
    for (size_t j = 0; j < k; ++j) {
        p[j] = _mm_mul_ps(p[j], f); // Multiply each SIMD vector by the factor
    }
}

template <typename T>
void Matrix<T>::add_row(size_t i, size_t j, T factor) {
    // Use SIMD intrinsics to add a scaled row to another row
    size_t k = cols_ / 4; // Number of SIMD vectors in a row
    __m128 f = _mm_set1_ps(factor); // Load the factor into a SIMD vector
    __m128 *p1 = (__m128 *)data_[i].data(); // Pointer to the first row data
    __m128 *p2 = (__m128 *)data_[j].data(); // Pointer to the second row data
    for (size_t l = 0; l < k; ++l) {
        p1[l] = _mm_add_ps(p1[l], _mm_mul_ps(p2[l], f)); // Add each scaled SIMD vector to the first row
    }
}



template<typename T>
Matrix<T> inverse(const Matrix<T> &mat) {
    // Check if the matrix is square, otherwise return an empty matrix
    if (mat.rows() != mat.cols()) {
        return Matrix<T>();
    }

    // Create a copy of the input matrix and an identity matrix of the same size
    Matrix<T> A(mat);
    Matrix<T> I = Matrix<T>::identity(mat.rows());

    // Get the number of rows and columns of the matrix
    int n = mat.rows();
    int m = mat.cols();

    // Loop over each column of the matrix
    for (int j = 0; j < m; j++) {
        // Find the pivot row with the largest element in the current column
        int pivot = j;
        T max = A(j, j);
        for (int i = j + 1; i < n; i++) {
            if (A(i, j) > max) {
                pivot = i;
                max = A(i, j);
            }
        }

        // Swap the pivot row with the current row
        if (pivot != j) {
            A.swap_rows(pivot, j);
            I.swap_rows(pivot, j);
        }

        // Divide the current row by the pivot element
        T inv_pivot = T(1) / A(j, j);
        A(j, j) = T(1); // set the pivot element to 1

        // Use SIMD intrinsics to perform vectorized operations on the current row
        __m256d inv_pivot_vec = _mm256_set1_pd(inv_pivot); // broadcast the inverse pivot to a vector
        __m256d *A_row_ptr = (__m256d *) &A(j, 0); // get a pointer to the current row of A
        __m256d *I_row_ptr = (__m256d *) &I(j, 0); // get a pointer to the current row of I
        int vec_size = m / 4; // get the number of vectorized operations

        // Loop over each vectorized operation
        for (int k = 0; k < vec_size; k++) {
            __m256d A_row_vec = loadu((const double*)A_row_ptr + k); // load a vector from the current row of A
            __m256d I_row_vec = loadu((const double*)I_row_ptr + k); // load a vector from the current row of I
            A_row_vec = _mm256_mul_pd(A_row_vec, inv_pivot_vec); // multiply the vector by the inverse pivot
            I_row_vec = _mm256_mul_pd(I_row_vec, inv_pivot_vec); // multiply the vector by the inverse pivot
            storeu((double*)A_row_ptr + k, A_row_vec); // store the vector back to the current row of A
            storeu((double*)I_row_ptr + k, I_row_vec); // store the vector back to the current row of I
        }

        // Handle the remaining elements that are not divisible by 4
        for (int k = vec_size * 4; k < m; k++) {
            A(j, k) *= inv_pivot;
            I(j, k) *= inv_pivot;
        }

        // Use OpenMP pragmas to parallelize the elimination process on the remaining rows
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            if (i != j) {
                T factor = A(i, j);
                A(i, j) = T(0); // set the eliminated element to 0

                // Use SIMD intrinsics to perform vectorized operations on the remaining rows
                __m256d factor_vec = _mm256_set1_pd(factor); // broadcast the factor to a vector
                __m256d *A_i_ptr = (__m256d *) &A(i, 0); // get a pointer to the current row of A
                __m256d *I_i_ptr = (__m256d *) &I(i, 0); // get a pointer to the current row of I
                __m256d *A_j_ptr = (__m256d *) &A(j, 0); // get a pointer to the pivot row of A
                __m256d *I_j_ptr = (__m256d *) &I(j, 0); // get a pointer to the pivot row of I

                // Loop over each vectorized operation
                for (int k = 0; k < vec_size; k++) {
                    __m256d A_i_vec = loadu((const double*)(A_i_ptr + k)); // load a vector from the current row of A
                    __m256d I_i_vec = loadu((const double*)(I_i_ptr + k)); // load a vector from the current row of I
                    __m256d A_j_vec = loadu((const double*)(A_j_ptr + k)); // load a vector from the pivot row of A
                    __m256d I_j_vec = loadu((const double*)(I_j_ptr + k)); // load a vector from the pivot row of I
                    __m256d factor_mul_A_j_vec = _mm256_mul_pd(factor_vec, A_j_vec); // factor * A(j,k)
                    __m256d factor_mul_I_j_vec = _mm256_mul_pd(factor_vec, I_j_vec); // factor * I(j,k)
                    A_i_vec = _mm256_sub_pd(A_i_vec, factor_mul_A_j_vec); // A(i,k) - factor * A(j,k)
                    I_i_vec = _mm256_sub_pd(I_i_vec, factor_mul_I_j_vec); // I(i,k) - factor * I(j,k)
                    storeu((double*)(A_i_ptr + k), A_i_vec); // store the vector back to the current row of A
                    storeu((double*)(I_i_ptr + k), I_i_vec); // store the vector back to the current row of I
                }

                // Handle the remaining elements that are not divisible by 4
                for (int k = vec_size * 4; k < m; k++) {
                    A(i, k) -= factor * A(j, k);
                    I(i, k) -= factor * I(j, k);
                }
            }
        }
    }
        return I;

}


template<typename T>
T determinant(const Matrix<T> &mat) {
    // Check if the matrix is square, otherwise return zero
    if (mat.rows() != mat.cols()) {
        return T(0);
    }

    // Create a copy of the input matrix
    Matrix<T> A(mat);

    // Get the number of rows and columns of the matrix
    int n = mat.rows();
    int m = mat.cols();

    // Initialize the determinant to one
    T det = T(1);

    // Loop over each column of the matrix
    for (int j = 0; j < m; j++) {
        // Find the pivot row with the largest element in the current column
        int pivot = j;
        T max = A(j, j);
        for (int i = j + 1; i < n; i++) {
            if (A(i, j) > max) {
                pivot = i;
                max = A(i, j);
            }
        }

        // Swap the pivot row with the current row
        if (pivot != j) {
            A.swap_rows(pivot, j);
            det *= -1; // update the sign of the determinant
        }

        // Divide the current row by the pivot element
        T inv_pivot = T(1) / A(j, j);
        A(j, j) = T(1); // set the pivot element to 1
        det *= max; // update the value of the determinant

        // Use SIMD intrinsics to perform vectorized operations on the current row
        __m256d inv_pivot_vec = _mm256_set1_pd(inv_pivot); // broadcast the inverse pivot to a vector
        __m256d* A_row_ptr = (__m256d*) &A(j, 0); // get a pointer to the current row of A
        int vec_size = m / 4; // get the number of vectorized operations

        // Loop over each vectorized operation
        for (int k = 0; k < vec_size; k++) {
            __m256d A_row_vec = loadu((const double*)A_row_ptr + k); // load a vector from the current row of A
            A_row_vec = _mm256_mul_pd(A_row_vec, inv_pivot_vec); // multiply the vector by the inverse pivot
            storeu((double*)A_row_ptr + k, A_row_vec); // store the vector back to the current row of A
        }

        // Handle the remaining elements that are not divisible by 4
        for (int k = vec_size * 4; k < m; k++) {
            A(j, k) *= inv_pivot;
        }

        // Use OpenMP pragmas to parallelize the elimination process on the remaining rows
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            if (i != j) {
                T factor = A(i, j);
                A(i, j) = T(0); // set the eliminated element to 0

                // Use SIMD intrinsics to perform vectorized operations on the remaining rows
                __m256d factor_vec = _mm256_set1_pd(factor); // broadcast the factor to a vector
                __m256d* A_i_ptr = (__m256d*) &A(i, 0); // get a pointer to the current row of A
                __m256d* A_j_ptr = (__m256d*) &A(j, 0); // get a pointer to the pivot row of A

                // Loop over each vectorized operation
                for (int k = 0; k < vec_size; k++) {
                    __m256d A_i_vec = loadu((const double*)(A_i_ptr + k)); // load a vector from the current row of A
                    __m256d A_j_vec = loadu((const double*)(A_j_ptr + k)); // load a vector from the pivot row of A
                    A_i_vec = _mm256_mul_pd(factor_vec, A_j_vec); // compute factor * A(j,k)
                    A_i_vec = _mm256_sub_pd(A_i_vec, A_i_vec); // compute A(i,k) - factor * A(j,k) and store in A(i,k)
                    storeu((double*)(A_i_ptr + k), A_i_vec); // store the vector back to the current row of A
                }

                // Handle the remaining elements that are not divisible by 4
                for (int k = vec_size * 4; k < m; k++) {
                    A(i, k) -= factor * A(j, k);
                }
            }
        }
    }

    return det;
}
#define UNTITLED3_MATRIX_H

#endif //UNTITLED3_MATRIX_H
