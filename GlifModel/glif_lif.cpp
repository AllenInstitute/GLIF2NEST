#include "glif_lif.h"

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


nest::RecordablesMap< allen::glif_lif >
  allen::glif_lif::recordablesMap_;

namespace nest
{
// Override the create() method with one call to RecordablesMap::insert_()
// for each quantity to be recorded.
template <>
void
RecordablesMap< allen::glif_lif >::create()
{
  // use standard names whereever you can for consistency!
  insert_( names::V_m, &allen::glif_lif::get_V_m_ );
}
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

allen::glif_lif::Parameters_::Parameters_()
  : th_inf_(0.0265) 
  , G_(4.6951e-09)
  , E_l_(-0.0774)
  , C_m_(9.9182e-11)
  , t_ref_(1.0)
  , V_reset_(0.0)
  //, v_init_(0.0)
  //: C_m( 250.0 )     // pF
  //, I_e( 0.0 )       // nA
  //, tau_syn( 2.0 )   // ms
  //, V_th( -55.0 )    // mV
  //, V_reset( -70.0 ) // mV
  //, t_ref( 2.0 )     // ms
{
}

allen::glif_lif::State_::State_( const Parameters_& p )
  : V_m_(0.0)
  , I_(0.0)
  //: V_m( p.V_reset )
  //, dI_syn( 0.0 )
  //, I_syn( 0.0 )
  //, I_ext( 0.0 )
  //, refr_count( 0 )
{
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
allen::glif_lif::Parameters_::get( DictionaryDatum& d ) const
{
  def<double>(d, names::V_th, th_inf_);
  def<double>(d, names::g, G_);
  def<double>(d, names::E_L, E_l_);
  def<double>(d, names::C_m, C_m_);
  def<double>(d, names::t_ref, t_ref_);
  def<double>(d, names::V_reset, V_reset_);
  //def<double>(d, names:v_init, v_init_);

  //( *d )[ names::C_m ] = C_m;
  //( *d )[ names::I_e ] = I_e;
  //( *d )[ names::tau_syn ] = tau_syn;
  //( *d )[ names::V_th ] = V_th;
  //( *d )[ names::V_reset ] = V_reset;
  //( *d )[ names::t_ref ] = t_ref;
}

void
allen::glif_lif::Parameters_::set( const DictionaryDatum& d )
{
  updateValue< double >(d, names::V_th, th_inf_ );
  updateValue< double >(d, names::g, G_ );
  updateValue< double >(d, names::E_L, E_l_ );
  updateValue< double >(d, names::C_m, C_m_ );
  updateValue< double >(d, names::t_ref, t_ref_ );
  updateValue< double >(d, names::V_reset, V_reset_ );


  //updateValue< double >( d, names::C_m, C_m );
  //updateValue< double >( d, names::I_e, I_e );
  //updateValue< double >( d, names::tau_syn, tau_syn );
  //updateValue< double >( d, names::V_th, th_inf_ );
  //updateValue< double >( d, names::V_reset, V_reset );
  //updateValue< double >( d, names::t_ref, t_ref );

}

void
allen::glif_lif::State_::get( DictionaryDatum& d ) const
{
  def< double >(d, names::V_m, V_m_ );

  //( *d )[ names::V_m ] = V_m;
}

void
allen::glif_lif::State_::set( const DictionaryDatum& d,
  const Parameters_& p )
{
  // Only the membrane potential can be set; one could also make other state
  // variables
  // settable.
  updateValue< double >( d, names::V_m, V_m_ );
}

allen::glif_lif::Buffers_::Buffers_( glif_lif& n )
  : logger_( n )
{
}

allen::glif_lif::Buffers_::Buffers_( const Buffers_&, glif_lif& n )
  : logger_( n )
{
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

allen::glif_lif::glif_lif()
  : Archiving_Node()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();
}

allen::glif_lif::glif_lif( const glif_lif& n )
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
allen::glif_lif::init_state_( const Node& proto )
{
  const glif_lif& pr = downcast< glif_lif >( proto );
  S_ = pr.S_;
}

void
allen::glif_lif::init_buffers_()
{
  B_.spikes_.clear();   // includes resize
  B_.currents_.clear(); // include resize
  B_.logger_.reset();  // includes resize
}

void
allen::glif_lif::calibrate()
{
  B_.logger_.init();

  V_.t_ref_remaining_ = 0.0;
  V_.t_ref_total_ = P_.t_ref_ * 1.0e-04;

  //const double h = Time::get_resolution().get_ms();
  //const double eh = std::exp( -h / P_.tau_syn );
  //const double tc = P_.tau_syn / P_.C_m;

  // compute matrix elements, all other elements 0
  //V_.P11 = eh;
  //V_.P22 = eh;
  //V_.P21 = h * eh;
  //V_.P30 = h / P_.C_m;
  //V_.P31 = tc * ( P_.tau_syn - ( h + P_.tau_syn ) * eh );
  //V_.P32 = tc * ( 1 - eh );
  // P33_ is 1

  // initial value ensure normalization to max amplitude 1.0
  //V_.pscInitialValue = 1.0 * numerics::e / P_.tau_syn;

  // refractory time in steps
  //V_.t_ref_steps = Time( Time::ms( P_.t_ref ) ).get_steps();
  //assert(
  //  V_.t_ref_steps >= 0 ); // since t_ref_ >= 0, this can only fail in error
}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void
allen::glif_lif::update( Time const& origin, const long from, const long to )
{
  
  const double dt = Time::get_resolution().get_ms() * 1.0e-04;
  //dt = dt*e-06;
  double v_old = S_.V_m_;

  for ( long lag = from; lag < to; ++lag )
  {

    if( V_.t_ref_remaining_ > 0.0)
    {
      // While neuron is in refractory period count-down in time steps (since dt
      // may change while in refractory) while holding the voltage at last peak.
      V_.t_ref_remaining_ -= dt;
      if( V_.t_ref_remaining_ < 0.0)
      {
        S_.V_m_ = P_.V_reset_;
      }
      else
      {
        S_.V_m_ = v_old;
      }
    }
    else
    {
      // Linear Euler forward (RK1) to find next V_m value
      S_.V_m_ = v_old + dt*(S_.I_ - P_.G_*(v_old - P_.E_l_))/P_.C_m_;
      
      if( S_.V_m_ > P_.th_inf_ ) 
      {
                //std::cout << spike_offset << std::endl;
        //Time t = Time::step( origin.get_steps() + lag + 1 );
        //std::cout << t.get_ms() << std::endl;
        //Time t_last = Time::step( origin.get_steps() + lag );
        //std::cout << t_last.get_ms() << std::endl;

        //std::cout << dt << std::endl;
        //std::cout << S_.V_m_ << std::endl;  
        V_.t_ref_remaining_ = V_.t_ref_total_;
        
        // Determine 
        double spike_offset = (1 - (P_.th_inf_ - v_old)/(S_.V_m_ - v_old)) * Time::get_resolution().get_ms();
        set_spiketime( Time::step( origin.get_steps() + lag + 1 ), spike_offset );
        SpikeEvent se;
        kernel().event_delivery_manager.send( *this, se, lag );
      }
    }

    S_.I_ = B_.currents_.get_value( lag );

    B_.logger_.record_data( origin.get_steps() + lag);

    v_old = S_.V_m_;
  }
}

void
allen::glif_lif::handle( SpikeEvent& e )
{
  assert( e.get_delay() > 0 );

  B_.spikes_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() );
}

void
allen::glif_lif::handle( CurrentEvent& e )
{
  assert( e.get_delay() > 0 );

  B_.currents_.add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ),
    e.get_weight() * e.get_current() );
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void
allen::glif_lif::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e ); // the logger does this for us
}
