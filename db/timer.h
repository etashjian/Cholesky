#ifndef _TIMER_H_
#define _TIMER_H_
#include <ctime>
#include <sys/time.h>

struct MyTimer {
    MyTimer() : _dirty(false) {};

    struct timeval start, stop;
    inline void startTimer() { gettimeofday( &start, NULL ); }
    inline void stopTimer() { gettimeofday( &stop, NULL ); }
    inline double elapsedInSec() {
        _curr = ( stop.tv_sec + stop.tv_usec * 1.0 / 1000000 ) - ( start.tv_sec + start.tv_usec * 1.0 / 1000000 );
        if( !_dirty ) {
            _first = _curr;
            _dirty = true;
        }
        return _curr;
    }
    inline int cummulativeSpeedup() {
        assert( _dirty );
        return static_cast<int>( _curr / _first );
    }

    bool _dirty;
    double _first, _prev, _curr;
};

#endif
