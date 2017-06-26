#ifndef GLIF_LIF_R_ASC_H
#define GLIF_LIF_R_ASC_H

#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

#include "dictdatum.h"

namespace allen
{

class glif_lif_r_asc : public nest::Archiving_Node
{
public:

  glif_lif_r_asc();

  glif_lif_r_asc( const glif_lif_r_asc& );

  using nest::Node::handle;
  using nest::Node::handles_test_event;

  nest::port send_test_event( nest::Node&, nest::port, nest::synindex, bool );

  void handle( nest::SpikeEvent& );
  void handle( nest::CurrentEvent& );
  void handle( nest::DataLoggingRequest& );

  nest::port handles_test_event( nest::SpikeEvent&, nest::port );
  nest::port handles_test_event( nest::CurrentEvent&, nest::port );
  nest::port handles_test_event( nest::DataLoggingRequest&, nest::port );

  bool is_off_grid() const  // uses off_grid events
  {
    return true;
  }

  void get_status( DictionaryDatum& ) const;
  void set_status( const DictionaryDatum& );

private:
  //! Reset parameters and state of neuron.

  //! Reset state of neuron.
  void init_state_( const Node& proto );

  //! Reset internal buffers of neuron.
  void init_buffers_();

  //! Initialize auxiliary quantities, leave parameters and state untouched.
  void calibrate();

  //! Take neuron through given time interval
  void update( nest::Time const&, const long, const long );

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap< glif_lif_r_asc >;
  friend class nest::UniversalDataLogger< glif_lif_r_asc >;


  struct Parameters_
  {
	double thr_init_; // initial threshold
	double th_inf_;  // infinity threshold
    //double V_th_;  // A constant spiking threshold
    double G_; // membrane conductance
    double E_l_; // resting potential
    double C_m_; // capacitance
    double t_ref_; // refractory time (ms)
    double V_reset_; // Membrane voltage following spike

    double a_spike_; // threshold additive constant following reset
    double b_spike_; //spike induced threshold
    double voltage_reset_a_; //voltage fraction following reset
    double voltage_reset_b_; // voltage additive constant following reset


    std::vector<double> asc_init_; // initial values of ASCurrents_
    std::vector<double> k_; // predefined time scale
    std::vector<double> asc_amps_;
    std::vector<double> r_;
    std::string V_dynamics_method_; // voltage dynamic methods

    Parameters_();

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum& );
  };


  struct State_
  {
    double V_m_;  // membrane potential
    std::vector<double> ASCurrents_; // after-spike currents
    double ASCurrents_sum_;

    double threshold_; // voltage threshold

    double I_; // external current

    State_( const Parameters_& );

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum&, const Parameters_& );
  };


  struct Buffers_
  {
    Buffers_( glif_lif_r_asc& );
    Buffers_( const Buffers_&, glif_lif_r_asc& );

    nest::RingBuffer spikes_;   //!< Buffer incoming spikes through delay, as sum
    nest::RingBuffer currents_; //!< Buffer incoming currents through delay,

    //! Logger for all analog data
    nest::UniversalDataLogger< glif_lif_r_asc > logger_;
  };

  struct Variables_
  {
    double t_ref_remaining_; // counter during refractory period, seconds
    double t_ref_total_; // total time of refractory period, seconds

    double last_spike_; // threshold spike component
    int method_; // voltage dynamics solver method flag: 0-linear forward euler; 1-linear exact
  };

  double get_V_m_() const
  {
    return S_.V_m_;
  }

  double get_AScurrents_sum_() const
  {
    return S_.ASCurrents_[0];
  }

  Parameters_ P_; 
  State_ S_;      
  Variables_ V_;  
  Buffers_ B_;    

  // Mapping of recordables names to access functions
  static nest::RecordablesMap< glif_lif_r_asc > recordablesMap_;
};

inline nest::port
allen::glif_lif_r_asc::send_test_event( nest::Node& target,
  nest::port receptor_type,
  nest::synindex,
  bool )
{
  nest::SpikeEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline nest::port
allen::glif_lif_r_asc::handles_test_event( nest::SpikeEvent&,
  nest::port receptor_type )
{
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  return 0;
}

inline nest::port
allen::glif_lif_r_asc::handles_test_event( nest::CurrentEvent&,
  nest::port receptor_type )
{
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  return 0;
}

inline nest::port
allen::glif_lif_r_asc::handles_test_event( nest::DataLoggingRequest& dlr,
  nest::port receptor_type )
{
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );

  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
glif_lif_r_asc::get_status( DictionaryDatum& d ) const
{
  // get our own parameter and state data
  P_.get( d );
  S_.get( d );

  // get information managed by parent class
  Archiving_Node::get_status( d );

  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
glif_lif_r_asc::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d, ptmp );   // throws if BadProperty

  Archiving_Node::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

} // namespace nest

#endif
