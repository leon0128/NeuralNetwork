#ifndef MATRIX_MATRIX_HPP
#define MATRIX_MATRIX_HPP

#include <ostream>
#include <stdexcept>
#include <cstddef>
#include <functional>

template<class T>
class Matrix
{
public:
    using ValueType = T;

    // create rowSize * columnSize matrix object
    Matrix();
    Matrix(std::size_t rowSize
        , std::size_t columnSize);
    Matrix(const Matrix<T>&);
    Matrix(Matrix<T>&&);
    Matrix &operator =(const Matrix<T> &other);
    Matrix &operator =(Matrix<T> &&other);
    ~Matrix();

    std::size_t row() const noexcept
        {return mRow;}
    std::size_t column() const noexcept
        {return mColumn;}
    ValueType *data() noexcept
        {return mData;}
    const ValueType *data() const noexcept
        {return mData;}

    void apply(const std::function<ValueType(ValueType)> &func);

    ValueType *operator [](std::size_t rowIndex)
        {return &data()[rowIndex * column()];}
    const ValueType *operator [](std::size_t rowIndex) const
        {return &data()[rowIndex * column()];}

    Matrix &operator +=(const Matrix &rhs);
    Matrix &operator +=(ValueType);
    Matrix &operator -=(const Matrix &rhs);
    Matrix &operator -=(ValueType);
    Matrix &operator *=(ValueType);
    Matrix &operator /=(ValueType);

private:
    std::size_t mRow; // row size
    std::size_t mColumn; // column size
    ValueType *mData; // raw data
};

template<class T>
Matrix<T> operator +(const Matrix<T> &lhs
    , const Matrix<T> &rhs);
template<class T>
Matrix<T> operator +(const Matrix<T> &lhs
    , typename Matrix<T>::ValueType rhs);
template<class T>
Matrix<T> operator +(typename Matrix<T>::ValueType lhs
    , const Matrix<T> &rhs);
template<class T>
Matrix<T> operator -(const Matrix<T> &lhs
    , const Matrix<T> &rhs);
template<class T>
Matrix<T> operator -(const Matrix<T> &lhs
    , typename Matrix<T>::ValueType rhs);
template<class T>
Matrix<T> operator -(typename Matrix<T>::ValueType lhs
    , const Matrix<T> &rhs);
template<class T>
Matrix<T> operator *(const Matrix<T> &lhs
    , const Matrix<T> &rhs);
template<class T>
Matrix<T> operator *(const Matrix<T> &lhs
    , typename Matrix<T>::ValueType rhs);
template<class T>
Matrix<T> operator *(typename Matrix<T>::ValueType lhs
    , const Matrix<T> &rhs);
template<class T>
Matrix<T> operator /(const Matrix<T> &lhs
    , typename Matrix<T>::ValueType rhs);
template<class T>
Matrix<T> operator /(typename Matrix<T>::ValueType lhs
    , const Matrix<T> &rhs);
template<class T>
Matrix<T> operator ~(const Matrix<T> &other);

template<class CharT
    , class TraitsT
    , class ValueType>
std::basic_ostream<CharT, TraitsT> &operator <<(std::basic_ostream<CharT, TraitsT> &os
    , const Matrix<ValueType> &matrix);

// implementations
template<class T>
Matrix<T>::Matrix()
    : mRow{0ull}
    , mColumn{0ull}
    , mData{nullptr}
{
}

template<class T>
Matrix<T>::Matrix(std::size_t rowSize
    , std::size_t columnSize)
    : mRow{rowSize}
    , mColumn{columnSize}
    , mData{mRow * mColumn != 0 ? new T[mRow * mColumn]{static_cast<T>(0)} : nullptr}
{
}

template<class T>
Matrix<T>::Matrix(const Matrix<T> &other)
    : mRow{other.row()}
    , mColumn{other.column()}
    , mData{mRow * mColumn != 0 ? new T[mRow * mColumn] : nullptr}
{
    for(std::size_t r{0ull}; r < row(); r++)
    {
        for(std::size_t c{0ull}; c < column(); c++)
            operator[](r)[c] = other[r][c];
    }
}

template<class T>
Matrix<T>::Matrix(Matrix<T> &&other)
    : mRow{other.row()}
    , mColumn{other.column()}
    , mData{other.data()}
{
    other.mRow = 0ull;
    other.mColumn = 0ull;
    other.mData = nullptr;
}

template<class T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
{
    if(this != &other)
    {
        delete []mData;
        mRow = other.row();
        mColumn = other.column();
        mData = mRow * mColumn != 0 ? new T[mRow * mColumn] : nullptr;

        for(std::size_t r{0ull}; r < mRow; r++)
            for(std::size_t c{0ull}; c < mColumn; c++)
                operator[](r)[c] = other[r][c];
    }

    return *this;
}

template<class T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other)
{
    if(this != &other)
    {
        delete []mData;
        mRow = other.row();
        mColumn = other.column();
        mData = other.data();

        other.mRow = 0ull;
        other.mColumn = 0ull;
        other.mData = nullptr;
    }

    return *this;
}

template<class T>
Matrix<T>::~Matrix()
{
    delete []mData;
}

template<class T>
void Matrix<T>::apply(const std::function<ValueType(ValueType)> &appliedFunction)
{
    for(std::size_t r{0ull}; r < row(); r++)
        for(std::size_t c{0ull}; c < column(); c++)
            operator[](r)[c] = appliedFunction(operator[](r)[c]);
}

template<class T>
Matrix<T> &Matrix<T>::operator +=(const Matrix<T> &rhs)
{
    if(row() != rhs.row()
        || column() != rhs.column())
        throw std::range_error("row or/and column sizes does not match.");

    for(std::size_t r{0ull}; r < row(); r++)
    {
        for(std::size_t c{0ull}; c < column(); c++)
            operator[](r)[c] += rhs[r][c];
    }

    return *this;
}

template<class T>
Matrix<T> &Matrix<T>::operator +=(ValueType v)
{
    for(std::size_t r{0ull}; r < row(); r++)
        for(std::size_t c{0ull}; c < column(); c++)
            operator[](r)[c] += v;

    return *this;
}

template<class T>
Matrix<T> &Matrix<T>::operator -=(const Matrix<T> &rhs)
{
    if(row() != rhs.row()
        || column() != rhs.column())
        throw std::range_error("row or/and column sizes does not match.");

    for(std::size_t r{0ull}; r < row(); r++)
    {
        for(std::size_t c{0ull}; c < column(); c++)
            operator[](r)[c] -= rhs[r][c];
    }

    return *this;
}

template<class T>
Matrix<T> &Matrix<T>::operator -=(ValueType v)
{
    for(std::size_t r{0ull}; r < row(); r++)
        for(std::size_t c{0ull}; c < column(); c++)
            operator[](r)[c] -= v;
        
    return *this;
}

template<class T>
Matrix<T> &Matrix<T>::operator *=(ValueType v)
{
    for(std::size_t r{0ull}; r < row(); r++)
        for(std::size_t c{0ull}; c < column(); c++)
            operator[](r)[c] *= v;
    
    return *this;
}

template<class T>
Matrix<T> &Matrix<T>::operator /=(ValueType v)
{
    for(std::size_t r{0ull}; r < row(); r++)
        for(std::size_t c{0ull}; c < column(); c++)
            operator[](r)[c] /= v;
    
    return *this;
}

template<class T>
Matrix<T> operator +(const Matrix<T> &lhs
    , const Matrix<T> &rhs)
{
    Matrix result{lhs};
    return result += rhs;
}

template<class T>
Matrix<T> operator +(const Matrix<T> &lhs
    , typename Matrix<T>::ValueType rhs)
{
    Matrix result{lhs};
    return result += rhs;
}

template<class T>
Matrix<T> operator +(typename Matrix<T>::ValueType lhs
    , const Matrix<T> &rhs)
{
    Matrix result{rhs};
    return result += lhs;
}

template<class T>
Matrix<T> operator -(const Matrix<T> &lhs
    , const Matrix<T> &rhs)
{
    Matrix result{lhs};
    return result -= rhs;
}

template<class T>
Matrix<T> operator -(const Matrix<T> &lhs
    , typename Matrix<T>::ValueType rhs)
{
    Matrix result{lhs};
    return result -= rhs;
}

template<class T>
Matrix<T> operator -(typename Matrix<T>::ValueType lhs
    , const Matrix<T> &rhs)
{
    Matrix result{rhs};
    for(std::size_t r{0ull}; r < result.row(); r++)
        for(std::size_t c{0ull}; c < result.column(); c++)
            result[r][c] = lhs - result[r][c];
    return result;
}

template<class T>
Matrix<T> operator *(const Matrix<T> &lhs
    , const Matrix<T> &rhs)
{
    if(lhs.column() != rhs.row())
        throw std::range_error("row or/and column sizes does not match.");
    
    Matrix<T> result{lhs.row(), rhs.column()};
    for(std::size_t resultRow{0ull}; resultRow < result.row(); resultRow++)
    {
        for(std::size_t resultColumn{0ull}; resultColumn < result.column(); resultColumn++)
        {
            for(std::size_t operandIndex{0ull}; operandIndex < lhs.column(); operandIndex++)
                result[resultRow][resultColumn] += lhs[resultRow][operandIndex] * rhs[operandIndex][resultColumn];
        }
    }

    return result;
}

template<class T>
Matrix<T> operator *(const Matrix<T> &lhs
    , typename Matrix<T>::ValueType rhs)
{
    Matrix<T> result{lhs};
    return result *= rhs;
}

template<class T>
Matrix<T> operator *(typename Matrix<T>::ValueType lhs
    , const Matrix<T> &rhs)
{
    Matrix<T> result{rhs};
    return result *= lhs;
}

template<class T>
Matrix<T> operator /(const Matrix<T> &lhs
    , typename Matrix<T>::ValueType rhs)
{
    Matrix<T> result{lhs};
    return result /= rhs;
}

template<class T>
Matrix<T> operator /(typename Matrix<T>::ValueType lhs
    , const Matrix<T> &rhs)
{
    Matrix<T> result{rhs};
    for(std::size_t r{0ull}; r < result.row(); r++)
        for(std::size_t c{0ull}; c < result.column(); c++)
            result[r][c] = lhs / result[r][c];
    
    return result;
}


template<class T>
Matrix<T> operator ~(const Matrix<T> &other)
{
    Matrix<T> result{other.column(), other.row()};

    for(std::size_t resultRow{0ull}; resultRow < result.row(); resultRow++)
    {
        for(std::size_t resultColumn{0ull}; resultColumn < result.column(); resultColumn++)
            result[resultRow][resultColumn] = other[resultColumn][resultRow];
    }

    return result;
}

template<class CharT
    , class TraitsT
    , class ValueType>
std::basic_ostream<CharT, TraitsT> &operator <<(std::basic_ostream<CharT, TraitsT> &os
    , const Matrix<ValueType> &matrix)
{
    os << '[';
    for(std::size_t r{0ull}; r < matrix.row(); r++)
    {
        os << '[';
        for(std::size_t c{0ull}; c < matrix.column(); c++)
            os << matrix[r][c] << (c + 1ull != matrix.column() ? "," : "");
        os << ']' << (r + 1ull != matrix.row() ? "\n" : "");
    }
    os << ']';

    return os;
}

#endif