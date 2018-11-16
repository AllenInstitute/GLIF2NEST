#ifndef GLIFMODULE_H
#define GLIFMODULE_H

#include "namedatum.h"

#include "slifunction.h"
#include "slimodule.h"

#include <string>

namespace nest
{

  class GlifModules : public SLIModule {
  public:
    GlifModules();
    ~GlifModules();

    void init(SLIInterpreter *i);

    const std::string name() const;

    const std::string commandstring() const;
    

  };
} // namespace glif

#endif
