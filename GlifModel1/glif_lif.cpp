#include "glif_lif.h"

using namespace nest;


nest::RecordablesMap<glif::glif_lif> glif::glif_lif::recordablesMap_;

namespace nest
{
  template<>
  void
  RecordablesMap<glif::glif_lif>::create() {
    insert_(names::V_m, &glif::glif_lif::get_V_m_);
  }
}

