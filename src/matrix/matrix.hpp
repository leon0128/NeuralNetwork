#ifndef MATRIX_MATRIX_HPP
#define MATRIX_MATRIX_HPP

#include <utility>
#include <ostream>
#include <stdexcept>
#include <cstddef>
#include <functional>
#include <valarray>

template<class T>
class Matrix
{
public:
    using ValueType = T;

    inline Matrix();
    inline Matrix(std::size_t rowSize
        , std::size_t columnSize
        , const ValueType &value = ValueType{});
    inline Matrix(std::size_t rowSize
        , std::size_t columnSize
        , const std::valarray<ValueType>&);
    inline Matrix(std::size_t rowSize
        , std::size_t columnSize
        , std::valarray<ValueType>&&);
    inline Matrix(const Matrix&);
    inline Matrix(Matrix&&);
    inline Matrix &operator =(const Matrix &other);
    inline Matrix &operator =(Matrix &&other);
    inline Matrix &operator =(const ValueType &val)
        {return mValarray = val, *this;}

    inline std::size_t row() const noexcept
        {return mRow;}
    inline std::size_t column() const noexcept
        {return mColumn;}

    inline ValueType operator()(std::size_t r, std::size_t c) const
        {return mValarray[r * mColumn + c];}
    inline ValueType &operator()(std::size_t r, std::size_t c)
        {return mValarray[r * mColumn + c];}

    inline Matrix operator +() const
        {return Matrix{mRow, mColumn, +mValarray};}
    inline Matrix operator -() const
        {return Matrix{mRow, mColumn, -mValarray};}
    inline Matrix operator ~() const
        {return Matrix{mColumn, mRow, mValarray[std::gslice(0, {mColumn, mRow}, {1, mColumn})]};}

    inline Matrix &operator +=(const Matrix &rhs)
        {return mValarray += rhs.mValarray, *this;}
    inline Matrix &operator +=(const ValueType &rhs)
        {return mValarray += rhs, *this;}
    inline Matrix &operator -=(const Matrix &rhs)
        {return mValarray -= rhs.mValarray, *this;}
    inline Matrix &operator -=(const ValueType &rhs)
        {return mValarray -= rhs, *this;}
    inline Matrix &operator *=(const Matrix &rhs)
        {return mValarray *= rhs.mValarray, *this;}
    inline Matrix &operator *=(const ValueType &rhs)
        {return mValarray *= rhs, *this;}
    inline Matrix &operator /=(const Matrix &rhs)
        {return mValarray /= rhs.mValarray, *this;}
    inline Matrix &operator /=(const ValueType &rhs)
        {return mValarray /= rhs, *this;}

    inline void swap(Matrix &other);
    inline ValueType sum() const
        {return mValarray.sum();}
    inline Matrix apply(ValueType func(ValueType)) const
        {return Matrix{mRow, mColumn, mValarray.apply(func)};}
    inline Matrix apply(ValueType func(const ValueType&)) const
        {return Matrix{mRow, mColumn, mValarray.apply(func)};}
    inline auto begin()
        {return std::begin(mValarray);}
    inline auto begin() const
        {return std::begin(mValarray);}
    inline auto end()
        {return std::end(mValarray);}
    inline auto end() const
        {return std::end(mValarray);}

    template<class U>
    friend inline void swap(Matrix<U>&, Matrix<U>&) noexcept;
    template<class CharT
        , class TraitsT
        , class U>
    friend inline std::basic_ostream<CharT, TraitsT> &operator <<(std::basic_ostream<CharT, TraitsT> &stream
        , const Matrix<U>&);
    
    template<class U>
    friend inline Matrix<U> operator +(const Matrix<U> &lhs
        , const Matrix<U> &rhs);
    template<class U>
    friend inline Matrix<U> operator +(const Matrix<U> &lhs
        , const typename Matrix<U>::ValueType &rhs);
    template<class U>
    friend inline Matrix<U> operator +(const typename Matrix<U>::ValueType &lhs
        , const Matrix<U> &rhs);
    template<class U>
    friend inline Matrix<U> operator -(const Matrix<U> &lhs
        , const Matrix<U> &rhs);
    template<class U>
    friend inline Matrix<U> operator -(const Matrix<U> &lhs
        , const typename Matrix<U>::ValueType &rhs);
    template<class U>
    friend inline Matrix<U> operator -(const typename Matrix<U>::ValueType &lhs
        , const Matrix<U> &rhs);
    template<class U>
    friend inline Matrix<U> operator *(const Matrix<U> &lhs
        , const Matrix<U> &rhs);
    template<class U>
    friend inline Matrix<U> operator *(const Matrix<U> &lhs
        , const typename Matrix<U>::ValueType &rhs);
    template<class U>
    friend inline Matrix<U> operator *(const typename Matrix<U>::ValueType &lhs
        , const Matrix<U> &rhs);
    template<class U>
    friend inline Matrix<U> operator /(const Matrix<U> &lhs
        , const Matrix<U> &rhs);
    template<class U>
    friend inline Matrix<U> operator /(const Matrix<U> &lhs
        , const typename Matrix<U>::ValueType &rhs);
    template<class U>
    friend inline Matrix<U> operator /(const typename Matrix<U>::ValueType &lhs
        , const Matrix<U> &rhs);

    template<class U>
    friend inline Matrix<U> matmul(const Matrix<U> &lhs
        , const Matrix<U> &rhs);
    template<class U>
    friend inline Matrix<U> exp(const Matrix<U>&);
    template<class U>
    friend inline Matrix<U> log(const Matrix<U>&);
    template<class U>
    friend inline Matrix<U> log10(const Matrix<U>&);
    template<class U>
    friend inline Matrix<U> pow(const Matrix<U> &base
        , const Matrix<U> &exp);
    template<class U>
    friend inline Matrix<U> pow(const Matrix<U> &base
        , const typename Matrix<U>::ValueType &exp);
    template<class U>
    friend inline Matrix<U> pow(const Matrix<U> &base
        , const typename Matrix<U>::ValueType &exp);
    template<class U>
    friend inline Matrix<U> sqrt(const Matrix<U>&);

private:
    std::size_t mRow; // row size
    std::size_t mColumn; // column size
    std::valarray<ValueType> mValarray;
};

template<class U>
inline void swap(Matrix<U>&, Matrix<U>&) noexcept;
template<class CharT
    , class TraitsT
    , class U>
inline std::basic_ostream<CharT, TraitsT> &operator <<(std::basic_ostream<CharT, TraitsT> &stream
    , const Matrix<U>&);

template<class U>
inline Matrix<U> operator +(const Matrix<U> &lhs
    , const Matrix<U> &rhs)
    {return Matrix{lhs} += rhs;}
template<class U>
inline Matrix<U> operator +(const Matrix<U> &lhs
    , const typename Matrix<U>::ValueType &rhs)
    {return Matrix{lhs} += rhs;}
template<class U>
inline Matrix<U> operator +(const typename Matrix<U>::ValueType &lhs
    , const Matrix<U> &rhs)
    {return Matrix{rhs.mRow, rhs.mColumn, lhs} += rhs;}
template<class U>
inline Matrix<U> operator -(const Matrix<U> &lhs
    , const Matrix<U> &rhs)
    {return Matrix{lhs} -= rhs;}
template<class U>
inline Matrix<U> operator -(const Matrix<U> &lhs
    , const typename Matrix<U>::ValueType &rhs)
    {return Matrix{lhs} -= rhs;}
template<class U>
inline Matrix<U> operator -(const typename Matrix<U>::ValueType &lhs
    , const Matrix<U> &rhs)
    {return Matrix{rhs.mRow, rhs.mColumn, lhs} -= rhs;}
template<class U>
inline Matrix<U> operator *(const Matrix<U> &lhs
    , const Matrix<U> &rhs)
    {return Matrix{lhs} *= rhs;}
template<class U>
inline Matrix<U> operator *(const Matrix<U> &lhs
    , const typename Matrix<U>::ValueType &rhs)
    {return Matrix{lhs} *= rhs;}
template<class U>
inline Matrix<U> operator *(const typename Matrix<U>::ValueType &lhs
    , const Matrix<U> &rhs)
    {return Matrix{rhs.mRow, rhs.mColumn, lhs} *= rhs;}
template<class U>
inline Matrix<U> operator /(const Matrix<U> &lhs
    , const Matrix<U> &rhs)
    {return Matrix{lhs} /= rhs;}
template<class U>
inline Matrix<U> operator /(const Matrix<U> &lhs
    , const typename Matrix<U>::ValueType &rhs)
    {return Matrix{lhs} /= rhs;}
template<class U>
inline Matrix<U> operator /(const typename Matrix<U>::ValueType &lhs
    , const Matrix<U> &rhs)
    {return Matrix{rhs.mRow, rhs.mColumn, lhs} /= rhs;}

template<class U>
inline Matrix<U> matmul(const Matrix<U> &lhs
    , const Matrix<U> &rhs);

template<class U>
inline Matrix<U> exp(const Matrix<U> &matrix)
    {return Matrix<U>{matrix.mRow, matrix.mColumn, std::exp(matrix.mValarray)};}

template<class U>
inline Matrix<U> log(const Matrix<U> &matrix)
    {return Matrix<U>{matrix.mRow, matrix.mColumn, std::log(matrix.mValarray)};}

template<class U>
inline Matrix<U> log10(const Matrix<U> &matrix)
    {return Matrix<U>{matrix.mRow, matrix.mColumn, std::log10(matrix.mValarray)};}

template<class U>
inline Matrix<U> pow(const Matrix<U> &base
    , const Matrix<U> &exp)
    {return Matrix<U>{base.mRow, base.mColumn, std::pow(base.mValarray, exp.mValarray)};}

template<class U>
inline Matrix<U> pow(const Matrix<U> &base
    , const typename Matrix<U>::ValueType &exp)
    {return Matrix<U>{base.mRow, base.mColumn, std::pow(base.mValarray, exp)};}

template<class U>
inline Matrix<U> pow(const typename Matrix<U>::ValueType &base
    , const Matrix<U> &exp)
    {return Matrix<U>{exp.mRow, exp.mColumn, std::pow(base, exp.mValarray)};}

template<class U>
inline Matrix<U> sqrt(const Matrix<U> &matrix)
    {return Matrix<U>{matrix.mRow, matrix.mColumn, std::sqrt(matrix.mValarray)};}

// implementations
template<class T>
inline Matrix<T>::Matrix()
    : mRow{0ull}
    , mColumn{0ull}
    , mValarray()
{
}

template<class T>
inline Matrix<T>::Matrix(std::size_t rowSize
    , std::size_t columnSize
    , const ValueType &value)
    : mRow{rowSize}
    , mColumn{columnSize}
    , mValarray(value, mRow * mColumn)
{
}

template<class T>
inline Matrix<T>::Matrix(std::size_t rowSize
    , std::size_t columnSize
    , const std::valarray<ValueType> &other)
    : mRow{rowSize}
    , mColumn{columnSize}
    , mValarray(other)
{
}

template<class T>
inline Matrix<T>::Matrix(std::size_t rowSize
    , std::size_t columnSize
    , std::valarray<ValueType> &&other)
    : mRow{rowSize}
    , mColumn{columnSize}
    , mValarray(other)
{
}

template<class T>
inline Matrix<T>::Matrix(const Matrix &other)
    : mRow{other.mRow}
    , mColumn{other.mColumn}
    , mValarray(other.mValarray)
{
}

template<class T>
inline Matrix<T>::Matrix(Matrix &&other)
    : mRow{other.mRow}
    , mColumn{other.mColumn}
    , mValarray(std::move(other.mValarray))
{
    other.mRow = 0ull;
    other.mColumn = 0ull;
}

template<class T>
inline Matrix<T> &Matrix<T>::operator=(const Matrix &other)
{
    if(this != &other)
    {
        mRow = other.mRow;
        mColumn = other.mColumn;
        mValarray = other.mValarray;
    }

    return *this;
}

template<class T>
inline Matrix<T> &Matrix<T>::operator=(Matrix &&other)
{
    if(this != &other)
    {
        mRow = other.mRow;
        mColumn = other.mColumn;
        mValarray = std::move(other.mValarray);

        other.mRow = 0ull;
        other.mColumn = 0ull;
    }

    return *this;
}

template<class T>
inline void Matrix<T>::swap(Matrix &other)
{
    std::swap(mRow, other.mRow);
    std::swap(mColumn, other.mColumn);
    std::swap(mValarray, other.mValarray);
}

template<class T>
inline void swap(Matrix<T> &lhs
    , Matrix<T> &rhs) noexcept
{
    std::swap(lhs.mRow, rhs.mRow);
    std::swap(lhs.mColumn, rhs.mColumn);
    std::swap(lhs.mValarray, rhs.mValarray);
}

template<class CharT
    , class TraitsT
    , class T>
inline std::basic_ostream<CharT, TraitsT> &operator <<(std::basic_ostream<CharT, TraitsT> &os
    , const Matrix<T> &matrix)
{
    os << '[';
    for(std::size_t r{0ull}; r < matrix.mRow; r++)
    {
        os << '[';
        for(std::size_t c{0ull}; c < matrix.mColumn; c++)
            os << matrix(r, c) << (c + 1ull != matrix.mColumn ? "," : "");
        os << ']';
    }
    os << ']';

    return os;
}

template<class U>
inline Matrix<U> matmul(const Matrix<U> &lhs
    , const Matrix<U> &rhs)
{
    std::size_t row{lhs.mRow};
    std::size_t column{rhs.mColumn};
    Matrix<U> result{row, column};
    for(std::size_t i{0ull}; i < row; i++)
    {
        for(std::size_t j{0ull}; j < column; j++)
            result(i, j)
                = (lhs.mValarray[std::slice(i * lhs.mColumn, lhs.mColumn, 1)]
                    * rhs.mValarray[std::slice(j, rhs.mRow, rhs.mColumn)])
                    .sum();
    }

    return result;
}

#endif