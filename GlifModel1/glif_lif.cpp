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

glif::glif_lif::Parameters_::Parameters_()
	: C_m_(1.359768920560577e-10) // Membrane capactiance
    , I_e_(0.0) // DC current,
    , G_(0.0) // membrance conductance
    , El_(0.0) // resting potential
    , V_reset_(0.0) // reset potential
    , V_th_(0.0) {

} 

glif::glif_lif::State_::State_(const glif::glif_lif::Parameters_ &p)
	: V_m_(p.El_) {

}

void glif::glif_lif::Parameters_::get(DictionaryDatum &d) const {
	def<double>(d, names::C_m, C_m_);
	def<double>(d, names::I_e, I_e_);
	def<double>(d, "G", G_);
	def<double>(d, "El", El_);
	def<double>(d, names::V_reset, V_reset_);
	def<double>(d, names::V_th, V_th_);
}

void glif::glif_lif::Parameters_::set(const DictionaryDatum& d) {
  updateValue< double >( d, names::C_m, C_m_);
  updateValue< double >( d, names::I_e, I_e_);
  updateValue< double >( d, "G", G_);
  updateValue< double >( d, names::V_th, V_th_);
  updateValue< double >( d, names::V_reset, V_reset_);
  updateValue< double >( d, "El", El_);
}

void glif::glif_lif::State_::get(DictionaryDatum &d) const {
	def<double>(d, names::V_m, V_m_);
}

void glif::glif_lif::State_::set(const DictionaryDatum &d, const Parameters_ &p) {
	updateValue<double>(d, names::V_m, V_m_);
}

glif::glif_lif::Buffers_::Buffers_(glif::glif_lif &n) 
	: logger_(n) {

}

glif::glif_lif::Buffers_::Buffers_(const Buffers_ &b, glif::glif_lif &n) 
	: logger_(n) {

}


glif::glif_lif::glif_lif() 
	: Archiving_Node()
	, P_()
	, S_(P_)
	, B_(*this) {
	recordablesMap_.create();
}

glif::glif_lif::glif_lif(const glif_lif &n)
	: Archiving_Node(n)
	, P_(n.P_)
	, S_(n.S_)
	, B_(n.B_, *this) {

}

void glif::glif_lif::init_state(const Node &proto) {
	const glif_lif &pr = downcast<glif_lif>(proto);
	S_ = pr.S_;
}

void glif::glif_lif::init_buffers_() {
	B_.spikes_.clear();
	B_.currents_.clear();
	B_.logger_.reset();
	Archiving_Node::clear_history();
}

void glif::glif_lif::calibrate() {
	B_.logger_.init();

	// TODO: implement actual calibration	
}

void glif::glif_lif::update(Time const& origin, const long from, const long to) {

	const double dt = Time::get_resolution().get_ms();

	for(long lag = from; lag < to; ++lag) {
		S_.V_m_ += dt*(1/P_.C_m_)*(P_.I_e_ - P_.G_*(S_.V_m_ - P_.El_));

		if(S_.V_m_ > P_.V_th_) {
			S_.V_m_ = P_.V_reset_;


		}



		B_.logger_.record_data(origin.get_steps() + lag);
	}
}