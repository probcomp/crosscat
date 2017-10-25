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
/////////////////////////////////////////////
// Creator : Donald Green dgreen@opcode-consulting.com
// Contributors :
// Description :
///////////////////////////////////////////

#include <ctime>
#include <cstring>
#include "DateTime.h"

Timer::Timer(bool reset)
{
    _start_t = mktime(NULL);
    if (reset) {
        Reset();
    }
}

void Timer::Reset()
{
    _start_t = get_time();
}

double Timer::GetElapsed()
{
    return difftime(get_time(), _start_t);
}

time_t Timer::get_time()
{
    return time(NULL);
}


bool Timer::Period(Timer &T, double *t, double period)
{
    if (T.GetElapsed() - *t > 0) {
        *t += period;
        return true;
    }
    return false;
}
