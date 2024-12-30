// Copyright (c) 2024

#ifndef CPP_LANG_H // Include guard
#define CPP_LANG_H

// must be C++11 or later
#if __cplusplus < 201103L
#   error "C++11 or later is required"
#endif

// [[deprecated]]
#if __cplusplus >= 201402L
#   define CPP_DEPRECATED [[deprecated]]
#else
#   define CPP_DEPRECATED [[gnu::deprecated]] 
#endif

// static_assert
#if __cpp_static_assert >= 201411L
#   define STATIC_ASSERT(x) static_assert(x, #x)
#else
#   define STATIC_ASSERT(x) static_assert(x, #x)
#endif

#endif
