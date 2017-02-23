#include "glifmodule.h"
#include <string>


#include "config.h"



#include "glif_lif.h"


// Includes from nestkernel:
#include "connection_manager_impl.h"
#include "connector_model_impl.h"
#include "dynamicloader.h"
#include "exceptions.h"
#include "genericmodel.h"
#include "genericmodel_impl.h"
#include "kernel_manager.h"
#include "model.h"
#include "model_manager_impl.h"
#include "nestmodule.h"
#include "target_identifier.h"

// Includes from sli:
#include "booldatum.h"
#include "integerdatum.h"
#include "sliexceptions.h"
#include "tokenarray.h"

#if defined( LTX_MODULE) | defined( LINKED_MODULE )
allen::GlifModules glifmodule_LTX_mod;
#endif

allen::GlifModules::GlifModules() {
#ifdef LINKED_MODULE
  nest::DynamicLoaderModule::registerLinkedModule(this);
#endif
}

allen::GlifModules::~GlifModules() {
}

const std::string allen::GlifModules::name() const {
  return std::string("Allen Institute Glif Modules");
}

const std::string
allen::GlifModules::commandstring() const {
  return std::string("(glifmodule-init) run");
}

void allen::GlifModules::init(SLIInterpreter *i) {
  nest::kernel().model_manager.register_node_model<glif_lif>("glif_lif");  
}
