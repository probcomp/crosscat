/*
 *   Copyright (c) 2010-2016, MIT Probabilistic Computing Project
 *
 *   Lead Developers: Dan Lovell and Jay Baxter
 *   Authors: Dan Lovell, Baxter Eaves, Jay Baxter, Vikash Mansinghka
 *   Research Leads: Vikash Mansinghka, Patrick Shafto
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */
#ifndef GUARD_CrossCat_Matrix_h
#define GUARD_CrossCat_Matrix_h

#include <algorithm>
#include <limits>
#include <stdexcept>

template<typename T>
class matrix
{
public:
    size_t size1() const
    {
        return _nrows;
    }
    size_t size2() const
    {
        return _ncols;
    }
    matrix() : _nrows(0), _ncols(0), _data(0) {}
    matrix(size_t nrows, size_t ncols)
        : _nrows(nrows), _ncols(ncols), _data(new T[nrows * ncols])
    {
        if (nrows > std::numeric_limits<size_t>::max() / ncols) {
            T *d = _data;
            _nrows = 0;
            _ncols = 0;
            _data = 0;
            delete[] d;
            throw std::bad_alloc();
        }
    }
    matrix(const matrix &m)
    {
        size_t i;
        _nrows = m._nrows;
        _ncols = m._ncols;
        _data = new T[_nrows * _ncols];
        for (i = 0; i < _nrows * _ncols; i++) {
            _data[i] = m._data[i];
        }
    }
    matrix &operator=(matrix m)
    {
        std::swap(_nrows, m._nrows);
        std::swap(_ncols, m._ncols);
        std::swap(_data, m._data);
        return *this;
    }
    ~matrix()
    {
        if (_data) {
            delete[] _data;
        }
    }
    T &operator()(size_t row, size_t col)
    {
        if (_nrows <= row) {
            throw std::range_error("row out of range");
        }
        if (_ncols <= col) {
            throw std::range_error("column out of range");
        }
        return _data[row * _ncols + col];
    }
    const T &operator()(size_t row, size_t col) const
    {
        if (_nrows <= row) {
            throw std::range_error("row out of range");
        }
        if (_ncols <= col) {
            throw std::range_error("column out of range");
        }
        return _data[row * _ncols + col];
    }
private:
    size_t _nrows;
    size_t _ncols;
    T *_data;
};

typedef matrix<double> MatrixD;

#endif // GUARD_CrossCat_Matrix_h
