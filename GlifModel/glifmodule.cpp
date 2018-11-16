#include "glifmodule.h"
#include <string>

#include "config.h"

#include "glif_lif.h"
#include "glif_lif_r.h"
#include "glif_lif_asc.h"
#include "glif_lif_r_asc.h"
#include "glif_lif_r_asc_a.h"
#include "glif_lif_psc.h"
#include "glif_lif_r_psc.h"
#include "glif_lif_asc_psc.h"
#include "glif_lif_r_asc_psc.h"
#include "glif_lif_r_asc_a_psc.h"
#include "glif_lif_cond.h"
#include "glif_lif_r_cond.h"
#include "glif_lif_asc_cond.h"
#include "glif_lif_r_asc_cond.h"
#include "glif_lif_r_asc_a_cond.h"

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
nest::GlifModules glifmodule_LTX_mod;
#endif

nest::GlifModules::GlifModules() {
#ifdef LINKED_MODULE
  nest::DynamicLoaderModule::registerLinkedModule(this);
#endif
}

nest::GlifModules::~GlifModules() {
}

const std::string nest::GlifModules::name() const {
  return std::string("Allen Institute Glif Modules");
}

const std::string
nest::GlifModules::commandstring() const {
  return std::string("(glifmodule-init) run");
}

void nest::GlifModules::init(SLIInterpreter *i) {
  nest::kernel().model_manager.register_node_model<glif_lif>("glif_lif");
  nest::kernel().model_manager.register_node_model<glif_lif_r>("glif_lif_r");
  nest::kernel().model_manager.register_node_model<glif_lif_asc>("glif_lif_asc");
  nest::kernel().model_manager.register_node_model<glif_lif_r_asc>("glif_lif_r_asc");
  nest::kernel().model_manager.register_node_model<glif_lif_r_asc_a>("glif_lif_r_asc_a");
  nest::kernel().model_manager.register_node_model<glif_lif_psc>("glif_lif_psc");
  nest::kernel().model_manager.register_node_model<glif_lif_r_psc>("glif_lif_r_psc");
  nest::kernel().model_manager.register_node_model<glif_lif_asc_psc>("glif_lif_asc_psc");
  nest::kernel().model_manager.register_node_model<glif_lif_r_asc_psc>("glif_lif_r_asc_psc");
  nest::kernel().model_manager.register_node_model<glif_lif_r_asc_a_psc>("glif_lif_r_asc_a_psc");
  nest::kernel().model_manager.register_node_model<glif_lif_cond>("glif_lif_cond");
  nest::kernel().model_manager.register_node_model<glif_lif_r_cond>("glif_lif_r_cond");
  nest::kernel().model_manager.register_node_model<glif_lif_asc_cond>("glif_lif_asc_cond");
  nest::kernel().model_manager.register_node_model<glif_lif_r_asc_cond>("glif_lif_r_asc_cond");
  nest::kernel().model_manager.register_node_model<glif_lif_r_asc_a_cond>("glif_lif_r_asc_a_cond");
}
