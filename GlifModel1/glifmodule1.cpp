#include "glifmodule1.h"
//

#include "config.h"

// Need to put this before glif_lif.h include b/c nestkernel/recordables_map can't find LiteralDatum
#include "glif_lif.h"

#include <string>

#include "kernel_manager.h"
#include "dynamicloader.h"
#include "model_manager.h"

#include "booldatum.h"
#include "integerdatum.h"
#include "sliexceptions.h"
#include "tokenarray.h"

#if defined( LTX_MODULE) | defined( LINKED_MODULE )
glif::GlifModule1 glifmodule1_LTX_mod;
#endif

glif::GlifModule1::GlifModule1() {
#ifdef LINKED_MODULE
  nest::DynamicLoaderModule::registerLinkedModule(this);
#endif
}

glif::GlifModule1::~GlifModule1() {
  // NOOP
}

const std::string glif::GlifModule1::name() const {
  return std::string("Glif Module One (LIF)");
}

const std::string
glif::GlifModule1::commandstring() const {
  return std::string("(glifmodule1-init) run");
}

void glif::GlifModule1::init(SLIInterpreter *i) {
  nest::kernel().model_manager.register_node_model<glif_lif>("glif_lif");  
}
