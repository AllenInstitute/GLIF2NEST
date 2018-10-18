#ifndef GLIF_LIF_R_PSC_H
#define GLIF_LIF_R_PSC_H

#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

#include "dictdatum.h"

namespace allen
{
/*
 * Author: Binghuang Cai, Kael Dai, Stefan Mihalas @ Allen Institute for Brain Science
 *
 */

class glif_lif_r_psc : public nest::Archiving_Node
{
public:

  glif_lif_r_psc();

  glif_lif_r_psc( const glif_lif_r_psc& );

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
  friend class nest::RecordablesMap< glif_lif_r_psc >;
  friend class nest::UniversalDataLogger< glif_lif_r_psc >;

  struct Parameters_
  {
    double th_inf_;  		// A threshold in mV
    double G_; 				// membrane conductance in nS
    double E_l_; 			// resting potential in mV
    double C_m_; 			// capacitance in pF
    double t_ref_; 			// refractory time in ms
    double a_spike_; 		// threshold additive constant following reset in mV
    double b_spike_; 		// spike induced threshold in 1/ms
    double voltage_reset_a_;// in 1/ms
    double voltage_reset_b_;// in 1/ms
    std::vector< double > tau_syn_; // synaptic port time constants in ms
    std::string V_dynamics_method_; // voltage dynamic methods

    // boolean flag which indicates whether the neuron has connections
    bool has_connections_;

    size_t n_receptors_() const; //!< Returns the size of tau_syn_

    Parameters_();

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum& );
  };


  struct State_
  {
    double V_m_;  // membrane potential in mV
    double threshold_; // voltage threshold in mV
    double I_; // external current in pA
    double I_syn_; 		// post synaptic current in pA

    std::vector< double > y1_; // synapse current evolution state 1 in pA
    std::vector< double > y2_; // synapse current evolution state 2 in pA

    State_( const Parameters_& );

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum&, const Parameters_& );
  };


  struct Buffers_
  {
    Buffers_( glif_lif_r_psc& );
    Buffers_( const Buffers_&, glif_lif_r_psc& );

    std::vector< nest::RingBuffer > spikes_;   //!< Buffer incoming spikes through delay, as sum
    nest::RingBuffer currents_; //!< Buffer incoming currents through delay,

    //! Logger for all analog data
    nest::UniversalDataLogger< glif_lif_r_psc > logger_;
  };

  struct Variables_
  {
    double t_ref_remaining_; 	// counter during refractory period, in ms
    double t_ref_total_; 		// total time of refractory period, in ms
  	double last_spike_; 		// last spike component of threshold in mV
    int method_; 				// voltage dynamics solver method flag: 0-linear forward euler; 1-linear exact

    std::vector< double > P11_; // synaptic current evolution parameter
    std::vector< double > P21_; // synaptic current evolution parameter
    std::vector< double > P22_; // synaptic current evolution parameter
    double P30_; 				// membrane current/voltage evolution parameter
    double P33_; 				// membrane voltage evolution parameter
    std::vector< double > P31_; // synaptic/membrane current evolution parameter
    std::vector< double > P32_; // synaptic/membrane current evolution parameter

    /** Amplitude of the synaptic current.
              This value is chosen such that a post-synaptic current with
              weight one has an amplitude of 1 pA.
    */
    std::vector< double > PSCInitialValues_; // post synaptic current intial values in pA

    unsigned int receptor_types_size_;
  };

  double get_V_m_() const
  {
    return S_.V_m_;
  }

  double get_I_syn_() const
  {
    return S_.I_syn_;
  }

  Parameters_ P_; //!< Free parameters.
  State_ S_;      //!< Dynamic state.
  Variables_ V_;  //!< Internal Variables
  Buffers_ B_;    //!< Buffers.

  //! Mapping of recordables names to access functions
  static nest::RecordablesMap< glif_lif_r_psc > recordablesMap_;

};

inline size_t
allen::glif_lif_r_psc::Parameters_::n_receptors_() const
{
  return tau_syn_.size();
}

inline nest::port
allen::glif_lif_r_psc::send_test_event( nest::Node& target,
  nest::port receptor_type,
  nest::synindex,
  bool )
{
  // You should usually not change the code in this function.
  // It confirms that the target of connection @c c accepts @c SpikeEvent on
  // the given @c receptor_type.
  nest::SpikeEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline nest::port
allen::glif_lif_r_psc::handles_test_event( nest::CurrentEvent&,
  nest::port receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c CurrentEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  return 0;
}

inline nest::port
allen::glif_lif_r_psc::handles_test_event( nest::DataLoggingRequest& dlr,
  nest::port receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c DataLoggingRequest on port 0.
  // The function also tells the built-in UniversalDataLogger that this node
  // is recorded from and that it thus needs to collect data during simulation.
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );

  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
glif_lif_r_psc::get_status( DictionaryDatum& d ) const
{
  // get our own parameter and state data
  P_.get( d );
  S_.get( d );

  // get information managed by parent class
  Archiving_Node::get_status( d );

  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
glif_lif_r_psc::set_status( const DictionaryDatum& d )
{
  Parameters_ ptmp = P_; // temporary copy in case of errors
  ptmp.set( d );         // throws if BadProperty
  State_ stmp = S_;      // temporary copy in case of errors
  stmp.set( d, ptmp );   // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  Archiving_Node::set_status( d );

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}

} // namespace

#endif 
