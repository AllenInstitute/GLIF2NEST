#ifndef GLIFMODULE1_H
#define GLIFMODULE1_H

#include "namedatum.h"

#include "slifunction.h"
#include "slimodule.h"

#include <string>

namespace glif
{

  class GlifModule1 : public SLIModule {
  public:
    GlifModule1();
    ~GlifModule1();

    void init(SLIInterpreter *i);

    const std::string name() const;

    const std::string commandstring() const;
    

  };
} // namespace glif

#endif
