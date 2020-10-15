#include "glif_lif_r_asc_a.h"

// C++ includes:
#include <limits>
#include <iostream>

// Includes from libnestutil:
#include "numerics.h"

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


nest::RecordablesMap< nest::glif_lif_r_asc_a >
  nest::glif_lif_r_asc_a::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template<>
void
RecordablesMap< nest::glif_lif_r_asc_a >::create()
{
  insert_( names::V_m, &nest::glif_lif_r_asc_a::get_V_m_ );
  insert_( Name("AScurrents_sum"), &nest::glif_lif_r_asc_a::get_AScurrents_sum_ );
}
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

nest::glif_lif_r_asc_a::Parameters_::Parameters_()
  : th_inf_(26.5) // in mV
  , G_(4.6951) // in nS
  , E_L_(-77.4) // in mv
  , C_m_(99.182) // in pF
  , t_ref_(0.5) // in ms
  , a_spike_(0.0) // in mV
  , b_spike_(0.0) // in 1/ms
  , voltage_reset_a_(0.0) // coefficient
  , voltage_reset_b_(0.0) // in mV
  , a_voltage_(0.0) // in 1/ms
  , b_voltage_(0.0) // in 1/ms
  , asc_init_(std::vector<double>(2, 0.0)) // in pA
  , k_(std::vector<double>(2, 0.0)) // in 1/ms
  , asc_amps_(std::vector<double>(2, 0.0)) // in pA
  , r_(std::vector<double>(2, 1.0)) // coefficient
  , V_dynamics_method_("linear_forward_euler")
{
}

nest::glif_lif_r_asc_a::State_::State_()
  : V_m_(-77.4) // in mV
  , ASCurrents_(std::vector<double>(2, 0.0)) // in pA
  , threshold_(26.5) // in mV
  , I_(0.0) // in pA
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
nest::glif_lif_r_asc_a::Parameters_::get( DictionaryDatum& d ) const
{
  def<double>(d, names::V_th, th_inf_);
  def<double>(d, names::g, G_);
  def<double>(d, names::E_L, E_L_);
  def<double>(d, names::C_m, C_m_);
  def<double>(d, names::t_ref, t_ref_);
  //def<double>(d, names::V_reset, V_reset_);

  def<double>(d, "a_spike", a_spike_);
  def<double>(d, "b_spike", b_spike_);
  def<double>(d, "a_reset", voltage_reset_a_);
  def<double>(d, "b_reset", voltage_reset_b_);

  def<double>(d, "a_voltage", a_voltage_);
  def<double>(d, "b_voltage", b_voltage_);

  def< std::vector<double> >(d, Name("asc_init"), asc_init_);
  def< std::vector<double> >(d, Name("k"), k_ );
  def< std::vector<double> >(d, Name("asc_amps"), asc_amps_);
  def< std::vector<double> >(d, Name("r"), r_);
  def<std::string>(d, "V_dynamics_method", V_dynamics_method_);

}

void
nest::glif_lif_r_asc_a::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >(d, names::V_th, th_inf_ );
  updateValue< double >(d, names::g, G_ );
  updateValue< double >(d, names::E_L, E_L_ );
  updateValue< double >(d, names::C_m, C_m_ );
  updateValue< double >(d, names::t_ref, t_ref_ );

  updateValue< double >(d, "a_spike", a_spike_ );
  updateValue< double >(d, "b_spike", b_spike_ );
  updateValue< double >(d, "a_reset", voltage_reset_a_ );
  updateValue< double >(d, "b_reset", voltage_reset_b_ );

  updateValue< double >(d, "a_voltage", a_voltage_ );
  updateValue< double >(d, "b_voltage", b_voltage_ );

  updateValue< std::vector<double> >(d, Name("asc_init"), asc_init_);
  updateValue< std::vector<double> >(d, Name("k"), k_ );
  updateValue< std::vector<double> >(d, Name("asc_amps"), asc_amps_);
  updateValue< std::vector<double> >(d, Name("r"), r_);
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

}

void
nest::glif_lif_r_asc_a::State_::get( DictionaryDatum& d ) const
{
  def< double >(d, names::V_m, V_m_ );
  def< std::vector<double> >(d, Name("ASCurrents"), ASCurrents_ );
}

void
nest::glif_lif_r_asc_a::State_::set( const DictionaryDatum& d,
  const Parameters_& p )
{
  updateValue< double >( d, names::V_m, V_m_ );
  updateValue< std::vector<double> >(d, Name("ASCurrents"), ASCurrents_ );

  V_m_ = p.E_L_;
  ASCurrents_ = p.asc_init_;
  threshold_ = p.th_inf_;
}

nest::glif_lif_r_asc_a::Buffers_::Buffers_( glif_lif_r_asc_a& n )
  : logger_( n )
{
}

nest::glif_lif_r_asc_a::Buffers_::Buffers_( const Buffers_&, glif_lif_r_asc_a& n )
  : logger_( n )
{
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

nest::glif_lif_r_asc_a::glif_lif_r_asc_a()
  : Archiving_Node()
  , P_()
  , S_()
  , B_( *this )
{
  recordablesMap_.create();
}

nest::glif_lif_r_asc_a::glif_lif_r_asc_a( const glif_lif_r_asc_a& n )
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
nest::glif_lif_r_asc_a::init_state_( const Node& proto )
{
  const glif_lif_r_asc_a& pr = downcast< glif_lif_r_asc_a >( proto );
  S_ = pr.S_;
}

void
nest::glif_lif_r_asc_a::init_buffers_()
{
  B_.spikes_.clear();   // includes resize
  B_.currents_.clear(); // include resize
  B_.logger_.reset();  // includes resize
}

void
nest::glif_lif_r_asc_a::calibrate()
{
  B_.logger_.init();

  V_.t_ref_remaining_ = 0.0;
  V_.t_ref_total_ = P_.t_ref_;
  V_.last_spike_ = 0.0;
  V_.last_voltage_ = 0.0;

  V_.method_ = 0; // default using linear forward euler for voltage dynamics
  if(P_.V_dynamics_method_=="linear_exact"){
     V_.method_ = 1;
  }

}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void
nest::glif_lif_r_asc_a::update( Time const& origin, const long from, const long to )
{
  const double dt = Time::get_resolution().get_ms();

  double v_old = S_.V_m_;
  double ASCurrents_old_sum = 0.0;
  double spike_component = 0.0;
  double voltage_component = 0.0;
  double th_old = S_.threshold_;
  double tau = P_.G_ / P_.C_m_;
  double exp_tau = std::exp(-tau * dt);

  for ( long lag = from; lag < to; ++lag )
  {

    // update threshold via exact solution of dynamics of spike component of threshold
    spike_component = V_.last_spike_ * std::exp(-P_.b_spike_ * dt);
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
        S_.V_m_ = P_.E_L_ + P_.voltage_reset_a_ * ( S_.V_m_ - P_.E_L_ ) + P_.voltage_reset_b_;

        // reset spike component of threshold
        V_.last_spike_ = V_.last_spike_ + P_.a_spike_;

        // rest the global threshold (voltage component of threshold: stay the same)
        S_.threshold_ = V_.last_spike_ + V_.last_voltage_ + P_.th_inf_;

        // Check if bad reset
        // TODO: Better way to handle?
        if(S_.V_m_ > S_.threshold_){
          printf("Simulation Terminated: Voltage (%f) reset above threshold (%f)!!\n", S_.V_m_, S_.threshold_);
        }
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

      // Calculate new ASCurrents value using exponential methods
      ASCurrents_old_sum = S_.ASCurrents_sum_;
      S_.ASCurrents_sum_ = 0.0;
      for(std::size_t a = 0; a < S_.ASCurrents_.size(); ++a)
      {
        S_.ASCurrents_sum_ += S_.ASCurrents_[a];
        S_.ASCurrents_[a] = S_.ASCurrents_[a] * std::exp(-P_.k_[a] * dt);
      }

      // voltage dynamic
      switch(V_.method_){
        // Linear Euler forward (RK1) to find next V_m value
        case 0: S_.V_m_ = v_old + dt*(S_.I_ + S_.ASCurrents_sum_ - P_.G_ * (v_old - P_.E_L_)) / P_.C_m_;
                break;
        // Linear Exact to find next V_m value
        case 1: S_.V_m_ = v_old * exp_tau + ((S_.I_+ S_.ASCurrents_sum_ + P_.G_ * P_.E_L_) / P_.C_m_) * (1 - exp_tau) / tau;
                break;
      }

      // Calculate exact voltage component of the threshold
      double beta = (S_.I_ + S_.ASCurrents_sum_ + P_.G_ * P_.E_L_) / P_.G_;
      double phi = P_.a_voltage_ / (P_.b_voltage_ - P_.G_ / P_.C_m_);
      voltage_component = phi * (v_old - beta) * std::exp(-P_.G_ * dt / P_.C_m_) + 1 / (std::exp(P_.b_voltage_ * dt))\
              * (V_.last_voltage_ - phi * (v_old - beta) - (P_.a_voltage_ / P_.b_voltage_) * (beta - P_.E_L_))\
              + (P_.a_voltage_ / P_.b_voltage_) * (beta - P_.E_L_);

      S_.threshold_ = V_.last_spike_ + voltage_component + P_.th_inf_;
      V_.last_voltage_ = voltage_component;

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

    // Update any external currents
    S_.I_ = B_.currents_.get_value( lag );

    // Save voltage
    B_.logger_.record_data( origin.get_steps() + lag);

    v_old = S_.V_m_;

    th_old = S_.threshold_;
  }
}

void
nest::glif_lif_r_asc_a::handle( SpikeEvent& e )
{
#ifdef GET_DELAY_STEPS
    assert(e.get_delay_steps() > 0);
#else
    assert(e.get_delay() > 0);
#endif

  B_.spikes_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() );
}

void
nest::glif_lif_r_asc_a::handle( CurrentEvent& e )
{
#ifdef GET_DELAY_STEPS
    assert(e.get_delay_steps() > 0);
#else
    assert(e.get_delay() > 0);
#endif

  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() * e.get_current() );
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void
nest::glif_lif_r_asc_a::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e ); // the logger does this for us
}
