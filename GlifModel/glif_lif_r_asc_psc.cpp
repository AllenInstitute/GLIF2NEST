#include "glif_lif_r_asc_psc.h"

// C++ includes:
#include <limits>
#include <iostream>

// Includes from libnestutil:
#include "numerics.h"
#include "propagator_stability.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"
#include "name.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "lockptrdatum.h"

using namespace nest;

nest::RecordablesMap< allen::glif_lif_r_asc_psc >
  allen::glif_lif_r_asc_psc::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template<>
void
RecordablesMap< allen::glif_lif_r_asc_psc >::create()
{
  insert_( names::V_m, &allen::glif_lif_r_asc_psc::get_V_m_ );
  insert_( Name("AScurrents_sum"), &allen::glif_lif_r_asc_psc::get_AScurrents_sum_ );
  insert_( names::I_syn, &allen::glif_lif_r_asc_psc::get_I_syn_ );
}
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

allen::glif_lif_r_asc_psc::Parameters_::Parameters_()
  : th_inf_(26.5) 			// in mV
  , G_(4.6951)				// in nS
  , E_l_(-77.4)				// in mV
  , C_m_(99.182)			// in pF
  , t_ref_(0.5)				// in ms
  , a_spike_(0.0)			// in mV
  , b_spike_(0.0)			// in 1/ms
  , voltage_reset_a_(0.0)	// coefficient
  , voltage_reset_b_(0.0)	// in mV
  , asc_init_(std::vector<double>(2, 0.0))	// in pA
  , k_(std::vector<double>(2, 0.0))			// in 1/ms
  , asc_amps_(std::vector<double>(2, 0.0))	// in pA
  , r_(std::vector<double>(2, 1.0))			// coefficient
  , tau_syn_(1, 2.0)		// in ms
  , V_dynamics_method_("linear_forward_euler")
  , has_connections_( false )
{
}

allen::glif_lif_r_asc_psc::State_::State_( const Parameters_& p )
  : V_m_(p.E_l_)	// in mV
  , ASCurrents_(p.asc_init_) // in pA
  , threshold_(p.th_inf_) // in mV
  , I_(0.0)		// in pA

{
	y1_.clear();
	y2_.clear();
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
allen::glif_lif_r_asc_psc::Parameters_::get( DictionaryDatum& d ) const
{
  def<double>(d, names::V_th, th_inf_);
  def<double>(d, names::g, G_);
  def<double>(d, names::E_L, E_l_);
  def<double>(d, names::C_m, C_m_);
  def<double>(d, names::t_ref, t_ref_);
  def<double>(d, "a_spike", a_spike_);
  def<double>(d, "b_spike", b_spike_);
  def<double>(d, "a_reset", voltage_reset_a_);
  def<double>(d, "b_reset", voltage_reset_b_);
  def< std::vector<double> >(d, Name("asc_init"), asc_init_);
  def< std::vector<double> >(d, Name("k"), k_ );
  def< std::vector<double> >(d, Name("asc_amps"), asc_amps_);
  def< std::vector<double> >(d, Name("r"), r_);
  ArrayDatum tau_syn_ad( tau_syn_ );
  def< ArrayDatum >( d, names::tau_syn, tau_syn_ad );
  def<std::string>(d, "V_dynamics_method", V_dynamics_method_);
  def< bool >( d, names::has_connections, has_connections_ );
}

void
allen::glif_lif_r_asc_psc::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >(d, names::V_th, th_inf_ );
  updateValue< double >(d, names::g, G_ );
  updateValue< double >(d, names::E_L, E_l_ );
  updateValue< double >(d, names::C_m, C_m_ );
  updateValue< double >(d, names::t_ref, t_ref_ );
  updateValue< double >(d, "a_spike", a_spike_ );
  updateValue< double >(d, "b_spike", b_spike_ );
  updateValue< double >(d, "a_reset", voltage_reset_a_ );
  updateValue< double >(d, "b_reset", voltage_reset_b_ );
  updateValue< std::vector<double> >(d, Name("asc_init"), asc_init_);
  updateValue< std::vector<double> >(d, Name("k"), k_ );
  updateValue< std::vector<double> >(d, Name("asc_amps"), asc_amps_);
  updateValue< std::vector<double> >(d, Name("r"), r_);
  updateValue< std::vector< double > >( d, "tau_syn", tau_syn_ );
  updateValue< std::string >(d, "V_dynamics_method", V_dynamics_method_);

  if ( C_m_ <= 0.0 )
  {
    throw BadProperty( "Capacitance must be strictly positive." );
  }

  if ( G_ <= 0.0 )
  {
    throw BadProperty( "Membrane conductance must be strictly positive." );
  }

  if ( t_ref_ <= 0.0 )
  {
    throw BadProperty( "Refractory time constant must be strictly positive." );
  }

  const size_t old_n_receptors = this->n_receptors_();
  if ( updateValue< std::vector< double > >( d, "tau_syn", tau_syn_ ) )
  {
    if ( this->n_receptors_() != old_n_receptors && has_connections_ == true )
    {
      throw BadProperty(
        "The neuron has connections, therefore the number of ports cannot be "
        "reduced." );
    }
    for ( size_t i = 0; i < tau_syn_.size(); ++i )
    {
      if ( tau_syn_[ i ] <= 0 )
      {
        throw BadProperty(
          "All synaptic time constants must be strictly positive." );
      }
    }
  }

}

void
allen::glif_lif_r_asc_psc::State_::get( DictionaryDatum& d ) const
{
  def< double >(d, names::V_m, V_m_ );
  def< std::vector<double> >(d, Name("ASCurrents"), ASCurrents_ );
}

void
allen::glif_lif_r_asc_psc::State_::set( const DictionaryDatum& d,
  const Parameters_& p )
{
  updateValue< double >( d, names::V_m, V_m_ );
  updateValue< std::vector<double> >(d, Name("ASCurrents"), ASCurrents_ );
}

allen::glif_lif_r_asc_psc::Buffers_::Buffers_( glif_lif_r_asc_psc& n )
  : logger_( n )
{
}

allen::glif_lif_r_asc_psc::Buffers_::Buffers_( const Buffers_&, glif_lif_r_asc_psc& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

allen::glif_lif_r_asc_psc::glif_lif_r_asc_psc()
  : Archiving_Node()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();
}

allen::glif_lif_r_asc_psc::glif_lif_r_asc_psc( const glif_lif_r_asc_psc& n )
  : Archiving_Node( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
allen::glif_lif_r_asc_psc::init_state_( const Node& proto )
{
  const glif_lif_r_asc_psc& pr = downcast< glif_lif_r_asc_psc >( proto );
  S_ = pr.S_;
}

void
allen::glif_lif_r_asc_psc::init_buffers_()
{
  B_.spikes_.clear();   // includes resize
  B_.currents_.clear(); // include resize
  B_.logger_.reset();  // includes resize
}

void
allen::glif_lif_r_asc_psc::calibrate()
{
  B_.logger_.init();

  V_.t_ref_remaining_ = 0.0;
  V_.t_ref_total_ = P_.t_ref_;
  V_.last_spike_ = 0.0;
  V_.method_ = 0; // default using linear forward euler for voltage dynamics
  if(P_.V_dynamics_method_=="linear_exact")
     V_.method_ = 1;

  // post synapse currents
  const double h = Time::get_resolution().get_ms(); // in second

  V_.P11_.resize( P_.n_receptors_() );
  V_.P21_.resize( P_.n_receptors_() );
  V_.P22_.resize( P_.n_receptors_() );
  V_.P31_.resize( P_.n_receptors_() );
  V_.P32_.resize( P_.n_receptors_() );

  S_.y1_.resize( P_.n_receptors_() );
  S_.y2_.resize( P_.n_receptors_() );
  V_.PSCInitialValues_.resize( P_.n_receptors_() );

  B_.spikes_.resize( P_.n_receptors_() );

  double Tau_ = P_.C_m_ / P_.G_;  // in second
  V_.P33_ = std::exp( -h / Tau_ );
  V_.P30_ = 1 / P_.C_m_ * ( 1 - V_.P33_ ) * Tau_;

  for (size_t i = 0; i < P_.n_receptors_() ; i++ )
  {
	double Tau_syn_s_ = P_.tau_syn_[i];  // in second
	// these P are independent
	V_.P11_[i] = V_.P22_[i] = std::exp( -h / Tau_syn_s_ );

	V_.P21_[i] = h * V_.P11_[i];

	// these are determined according to a numeric stability criterion
	// input time parameter shall be in ms, capacity in pF
	V_.P31_[i] = propagator_31( P_.tau_syn_[i], Tau_, P_.C_m_, h );
	V_.P32_[i] = propagator_32( P_.tau_syn_[i], Tau_, P_.C_m_, h );

	V_.PSCInitialValues_[i] = 1.0 * numerics::e / Tau_syn_s_;
	B_.spikes_[ i ].resize();
  }

}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void
allen::glif_lif_r_asc_psc::update( Time const& origin, const long from, const long to )
{ 
  const double dt = Time::get_resolution().get_ms();

  double v_old = S_.V_m_;
  double spike_component = 0.0;
  double th_old=S_.threshold_;

  for ( long lag = from; lag < to; ++lag )
  {

	// update threshold via exact solution of dynamics of spike component of threshold
	spike_component = V_.last_spike_ * std::exp(-P_.b_spike_ * dt);
	S_.threshold_ = spike_component + P_.th_inf_;
	V_.last_spike_ = spike_component;

    if( V_.t_ref_remaining_ > 0.0)
    {
      // While neuron is in refractory period count-down in time steps (since dt
      // may change while in refractory) while holding the voltage at last peak.
      V_.t_ref_remaining_ -= dt;
      if( V_.t_ref_remaining_ <= 0.0)
      {
        // Neuron has left refractory period, reset voltage and after-spike current

	    // Reset ASC_currents
      	for(std::size_t a = 0; a < S_.ASCurrents_.size(); ++a)
      	{
      		S_.ASCurrents_[a] = P_.asc_amps_[a] + S_.ASCurrents_[a] * P_.r_[a] * std::exp(-P_.k_[a] * V_.t_ref_total_);
      	}

      	// Reset voltage
        S_.V_m_ = P_.E_l_ + P_.voltage_reset_a_ * ( S_.V_m_ - P_.E_l_ ) + P_.voltage_reset_b_;

        // reset spike component of threshold
        V_.last_spike_ = V_.last_spike_ + P_.a_spike_;
        S_.threshold_ = V_.last_spike_ + P_.th_inf_;

        // Check if bad reset
        // TODO: Better way to handle?
        if(S_.V_m_ > S_.threshold_) printf("Simulation Terminated: Voltage (%f) reset above threshold (%f)!!\n", S_.V_m_, S_.threshold_);
        assert( S_.V_m_ <= S_.threshold_ );

      }
      else
      {
        S_.V_m_ = v_old;
      }
    }
    else
    {
      // Integrate voltage and currents

      // Calculate new ASCurrents value using expoential methods
      S_.ASCurrents_sum_ = 0.0;
      for(std::size_t a = 0; a < S_.ASCurrents_.size(); ++a)
      {
      	S_.ASCurrents_sum_ += S_.ASCurrents_[a];
      	S_.ASCurrents_[a] = S_.ASCurrents_[a] * std::exp(-P_.k_[a] * dt);
      }

      // voltage dynamics of membranes
      switch(V_.method_){
        // Linear Euler forward (RK1) to find next V_m value
        case 0: S_.V_m_ = v_old + dt*(S_.I_ + S_.ASCurrents_sum_ - P_.G_* (v_old - P_.E_l_))/P_.C_m_;
        		break;
        // Linear Exact to find next V_m value
        case 1: S_.V_m_ = v_old * V_.P33_ + (S_.I_ + S_.ASCurrents_sum_ + P_.G_ * P_.E_l_) * V_.P30_;
        		break;
      }

      // add synapse component for voltage dynamics
      S_.I_syn_ = 0.0;
      for ( size_t i = 0; i < P_.n_receptors_(); i++ )
      {
        S_.V_m_ += V_.P31_[i] * S_.y1_[i] + V_.P32_[i] * S_.y2_[i];
        S_.I_syn_ += S_.y2_[i];
      }

      // Check if there is an action potential
      if( S_.V_m_ >  S_.threshold_ )
      {
	    // Marks that the neuron is in a refractory period
        V_.t_ref_remaining_ = V_.t_ref_total_;

	    // Find the exact time during this step that the neuron crossed the threshold and record it
        double spike_offset = (1 - (v_old - th_old)/(( S_.threshold_- th_old)-(S_.V_m_ - v_old))) * Time::get_resolution().get_ms();
        set_spiketime( Time::step( origin.get_steps() + lag + 1 ), spike_offset );
        SpikeEvent se;
        se.set_offset(spike_offset);
        kernel().event_delivery_manager.send( *this, se, lag );
      }
    }

    // alpha shape PSCs
    for( size_t i = 0; i < P_.n_receptors_(); i++ )
    {
      S_.y2_[i] = V_.P21_[i] * S_.y1_[i] + V_.P22_[i] * S_.y2_[i];
      S_.y1_[i] *= V_.P11_[i];

      // Apply spikes delivered in this step: The spikes arriving at T+1 have an
      // immediate effect on the state of the neuron
      S_.y1_[i] += V_.PSCInitialValues_[i] * B_.spikes_[i].get_value( lag );
    }

    // Update any external currents
    S_.I_ = B_.currents_.get_value( lag );

    // Save voltage
    B_.logger_.record_data( origin.get_steps() + lag);

    v_old = S_.V_m_;

    th_old = S_.threshold_;
  }
}

nest::port
allen::glif_lif_r_asc_psc::handles_test_event( SpikeEvent&,
  rport receptor_type )
{
  if ( receptor_type <= 0
    || receptor_type > static_cast< port >( P_.n_receptors_() ) )
  {
    throw IncompatibleReceptorType( receptor_type, get_name(), "SpikeEvent" );
  }

  P_.has_connections_ = true;
  return receptor_type;
}

void
allen::glif_lif_r_asc_psc::handle( SpikeEvent& e )
{
  assert( e.get_delay() > 0 );

  B_.spikes_[e.get_rport() - 1].add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() * e.get_multiplicity()  );
}

void
allen::glif_lif_r_asc_psc::handle( CurrentEvent& e )
{
  assert( e.get_delay() > 0 );

  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() * e.get_current() );
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void
allen::glif_lif_r_asc_psc::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e ); // the logger does this for us
}
