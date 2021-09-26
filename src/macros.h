/* Copyright 2020 Barcelona Supercomputing Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#ifndef _MACROS_H_
#define _MACROS_H_

#ifdef ENABLE_HWC_VE // If ENABLE_HWC_VE is defined PMC will be used, otherwise wont have any effect.
    #include "hw_counters_ve.h"
  #define HW_COUNTERS(x) x 
#else
  #define HW_COUNTERS(x)
#endif

#ifdef DEBUG
  #define DEBUG_MSG(x) x
#else
  #define DEBUG_MSG(x)
#endif


#endif
