
#ifndef UNTITLED3_VECTOR_H
// A class template for vectors of arbitrary size and type
template<typename T>
class Vector {
private:
    // The size of the vector
    size_t size_;
    // The data of the vector stored in a dynamic array
    T *data_;
public:
    // Default constructor that creates an empty vector
    Vector();

    // Constructor that creates a vector of a given size and fills it with a default value
    Vector(size_t size, const T &value = T());

    // Constructor that creates a vector from an initializer list
    Vector(std::initializer_list<T> list);

    // Copy constructor that creates a deep copy of another vector
    Vector(const Vector<T> &other);

    // Move constructor that transfers ownership of another vector's data
    Vector(Vector<T> &&other);

    // Destructor that frees the allocated memory
    ~Vector();

    // Copy assignment operator that creates a deep copy of another vector
    Vector<T> &operator=(const Vector<T> &other);

    // Move assignment operator that transfers ownership of another vector's data
    Vector<T> &operator=(Vector<T> &&other);

    // Subscript operator that returns a reference to an element at a given index
    T &operator[](size_t index);

    // Subscript operator that returns a const reference to an element at a given index
    const T &operator[](size_t index) const;

    // Unary plus operator that returns a copy of the vector
    Vector<T> operator+() const;

    // Unary minus operator that returns the negation of the vector
    Vector<T> operator-() const;

    // Binary plus operator that returns the sum of two vectors
    Vector<T> operator+(const Vector<T> &other) const;

    // Binary minus operator that returns the difference of two vectors
    Vector<T> operator-(const Vector<T> &other) const;

    // Scalar multiplication operator that returns the product of a vector and a scalar
    template<typename U>
    friend Vector<U> operator*(const U &scalar, const Vector<U> &vec);

    template<typename U>
    friend Vector<U> operator*(const Vector<U> &vec, const U &scalar);

    // Scalar division operator that returns the quotient of a vector and a scalar
    template<typename U>
    friend Vector<U> operator/(const Vector<U> &vec, const U &scalar);

    // Dot product operator that returns the scalar product of two vectors
    T operator*(const Vector<T> &other) const;

    // Compound assignment operators that modify the vector in place
    Vector<T> &operator+=(const Vector<T> &other);

    Vector<T> &operator-=(const Vector<T> &other);

    template<typename U>
    Vector<T> &operator*=(const U &scalar);

    template<typename U>
    Vector<T> &operator/=(const U &scalar);

    // Equality operator that returns true if two vectors have the same size and elements
    bool operator==(const Vector<T> &other) const;

    // Inequality operator that returns true if two vectors are not equal
    bool operator!=(const Vector<T> &other) const;

    // Stream insertion operator that prints the vector to an output stream
    template<typename U>
    friend std::ostream &operator<<(std::ostream &os, const Vector<U> &vec);

    // Stream extraction operator that reads the vector from an input stream
    template<typename U>
    friend std::istream &operator>>(std::istream &is, Vector<U> &vec);

    // A function that returns the size of the vector
    size_t size() const;

    // A function that returns a pointer to the data of the vector
    T *data();

    // A function that returns a const pointer to the data of the vector
    const T *data() const;
};


// Vector class implementation

// Default constructor
template<typename T>
Vector<T>::Vector() : size_(0), data_(nullptr) {}

// Constructor with size and default value
template<typename T>
Vector<T>::Vector(size_t size, const T &value) : size_(size), data_(new T[size]) {
    std::fill_n(data_, size, value);
}

// Constructor with initializer list
template<typename T>
Vector<T>::Vector(std::initializer_list<T> list) : size_(list.size()), data_(new T[size_]) {
    std::copy(list.begin(), list.end(), data_);
}

// Copy constructor
template<typename T>
Vector<T>::Vector(const Vector<T> &other) : size_(other.size_), data_(new T[size_]) {
    std::copy(other.data_, other.data_ + size_, data_);
}

// Move constructor
template<typename T>
Vector<T>::Vector(Vector<T> &&other) : size_(other.size_), data_(other.data_) {
    other.size_ = 0;
    other.data_ = nullptr;
}

// Destructor
template<typename T>
Vector<T>::~Vector() {
    delete[] data_;
}

// Copy assignment operator
template<typename T>
Vector<T> &Vector<T>::operator=(const Vector<T> &other) {
    if (this != &other) {
        delete[] data_;
        size_ = other.size_;
        data_ = new T[size_];
        std::copy(other.data_, other.data_ + size_, data_);
    }
    return *this;
}

// Move assignment operator
template<typename T>
Vector<T> &Vector<T>::operator=(Vector<T> &&other) {
    if (this != &other) {
        delete[] data_;
        size_ = other.size_;
        data_ = other.data_;
        other.size_ = 0;
        other.data_ = nullptr;
    }
    return *this;
}

// Subscript operator
template<typename T>
T &Vector<T>::operator[](size_t index) {
    if (index >= size_) {
        throw std::out_of_range("Vector index out of range");
    }
    return data_[index];
}

// Const subscript operator
template<typename T>
const T &Vector<T>::operator[](size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("Vector index out of range");
    }
    return data_[index];
}

// Unary plus operator
template<typename T>
Vector<T> Vector<T>::operator+() const {
    return *this;
}

// Unary minus operator
template<typename T>
Vector<T> Vector<T>::operator-() const {
    Vector<T> result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result[i] = -data_[i];
    }
    return result;
}

// Binary plus operator
template<typename T>
Vector<T> Vector<T>::operator+(const Vector<T> &other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector sizes do not match for addition");
    }
    Vector<T> result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result[i] = data_[i] + other[i];
    }
    return result;
}

// Binary minus operator
template<typename T>
Vector<T> Vector<T>::operator-(const Vector<T> &other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector sizes do not match for subtraction");
    }
    Vector<T> result(size_);
    for (size_t i = 0; i < size_; ++i) {
        result[i] = data_[i] - other[i];
    }
    return result;
}

// Scalar multiplication operator
template<typename U>
Vector<U> operator*(const U &scalar, const Vector<U> &vec) {
    Vector<U> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] * scalar;
    }
    return result;
}


template<typename U>
Vector<U> operator*(const Vector<U> &vec, const U &scalar) {
    return scalar * vec;
}

// Scalar division operator
template<typename U>
Vector<U> operator/(const Vector<U> &vec, const U &scalar) {
    Vector<U> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] / scalar;
    }
    return result;
}


// Dot product operator
template<typename T>
T Vector<T>::operator*(const Vector<T> &other) const {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector sizes do not match for dot product");
    }
    T result = T();
    for (size_t i = 0; i < size_; ++i) {
        result += data_[i] * other[i];
    }
    return result;
}

// Compound assignment operators
template<typename T>
Vector<T> &Vector<T>::operator+=(const Vector<T> &other) {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector sizes do not match for addition");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] += other[i];
    }
    return *this;
}

template<typename T>
Vector<T> &Vector<T>::operator-=(const Vector<T> &other) {
    if (size_ != other.size_) {
        throw std::invalid_argument("Vector sizes do not match for subtraction");
    }
    for (size_t i = 0; i < size_; ++i) {
        data_[i] -= other[i];
    }
    return *this;
}

template<typename T>
template<typename U>
Vector<T> &Vector<T>::operator*=(const U &scalar) {
    for (size_t i = 0; i < size_; ++i) {
        data_[i] *= scalar;
    }
    return *this;
}

template<typename T>
template<typename U>
Vector<T> &Vector<T>::operator/=(const U &scalar) {
    for (size_t i = 0; i < size_; ++i) {
        data_[i] /= scalar;
    }
    return *this;
}

// Equality operator
template<typename T>
bool Vector<T>::operator==(const Vector<T> &other) const {
    if (size_ != other.size_) {
        return false;
    }
    for (size_t i = 0; i < size_; ++i) {
        if (data_[i] != other[i]) {
            return false;
        }
    }
    return true;
}

// Inequality operator
template<typename T>
bool Vector<T>::operator!=(const Vector<T> &other) const {
    return !(*this == other);
}

// Stream insertion operator
template<typename T>
std::ostream &operator<<(std::ostream &os, const Vector<T> &vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

// Stream extraction operator
template<typename T>
std::istream &operator>>(std::istream &is, Vector<T> &vec) {
    T value;
    for (size_t i = 0; i < vec.size(); ++i) {
        is >> value;
        vec[i] = value;
    }
    return is;
}

// A function that returns the size of the vector
template<typename T>
size_t Vector<T>::size() const {
    return size_;
}

// A function that returns a pointer to the data of the vector
template<typename T>
T *Vector<T>::data() {
    return data_;
}

// A function that returns a const pointer to the data of the vector
template<typename T>
const T *Vector<T>::data() const {
    return data_;
}
#define UNTITLED3_VECTOR_H

#endif //UNTITLED3_VECTOR_H
