#ifndef GLIF_LIF_PSC_H
#define GLIF_LIF_PSC_H

#include "archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"

#include "dictdatum.h"

/* BeginDocumentation
Name: glif_lif_psc - Generalized leaky integrate and fire (GLIF) model 1 -
                     Traditional leaky integrate and fire (LIF) model.

Description:

  glif_lif_psc is an implementation of a generalized leaky integrate and fire (GLIF) model 1
  (i.e., traditional leaky integrate and fire (LIF) model) [1] with alpha-function shaped
  synaptic currents. Incoming spike events induce a post-synaptic change of current modeled
  by an alpha function. The alpha function is normalized such that an event of weight 1.0
  results in a peak current of 1 pA at t = tau_syn. On the postsynapic side,
  there can be arbitrarily many synaptic time constants. This can be reached by specifying
  separate receptor ports, each for a different time constant.
  The port number has to match the respective "receptor_type" in the connectors.

Parameters:

  The following parameters can be set in the status dictionary.

  V_m               double - Membrane potential in mV
  V_th              double - Instantaneous threshold in mV.
  g                 double - Membrane conductance in nS.
  E_L               double - Resting membrane potential in mV.
  C_m               double - Capacitance of the membrane in pF.
  t_ref             double - Duration of refractory time in ms.
  V_reset           double - Reset potential of the membrane in mV.
  tau_syn           double vector - Rise time constants of the synaptic alpha function in ms.
  V_dynamics_method string - Voltage dynamics (Equation (1) in [1]) solution methods:
                             'linear_forward_euler' - Linear Euler forward (RK1) to find next V_m value, or
                             'linear_exact' - Linear exact to find next V_m value.

References:
  [1] Teeter C, Iyer R, Menon V, Gouwens N, Feng D, Berg J, Szafer A,
      Cain N, Zeng H, Hawrylycz M, Koch C, & Mihalas S (2018)
      Generalized leaky integrate-and-fire models classify multiple neuron types.
      Nature Communications 9:709.

Author: Binghuang Cai and Kael Dai @ Allen Institute for Brain Science
*/

namespace nest
{

class glif_lif_psc : public nest::Archiving_Node
{
public:

  glif_lif_psc();

  glif_lif_psc( const glif_lif_psc& );

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
  friend class nest::RecordablesMap< glif_lif_psc >;
  friend class nest::UniversalDataLogger< glif_lif_psc >;


  struct Parameters_
  {
    double th_inf_; // A constant spiking threshold in mV
    double G_; // membrane conductance in nS
    double E_L_; // resting potential in mV
    double C_m_; // capacitance in pF
    double t_ref_; // refractory time in ms
    double V_reset_; // Membrane voltage following spike in mV
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
    double V_m_; // membrane potential in mV
    double I_; // external current in pA
    double I_syn_; // post synaptic current in pA
    std::vector< double > y1_; // synapse current evolution state 1 in pA
    std::vector< double > y2_; // synapse current evolution state 2 in pA

    State_();

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum&, const Parameters_& );
  };


  struct Buffers_
  {
    Buffers_( glif_lif_psc& );
    Buffers_( const Buffers_&, glif_lif_psc& );

    std::vector< nest::RingBuffer > spikes_;   //!< Buffer incoming spikes through delay, as sum
    nest::RingBuffer currents_; //!< Buffer incoming currents through delay,

    //! Logger for all analog data
    nest::UniversalDataLogger< glif_lif_psc > logger_;
  };

  struct Variables_
  {
    double t_ref_remaining_; // counter during refractory period, in ms
    double t_ref_total_; // total time of refractory period, in ms
    int method_; // voltage dynamics solver method flag: 0-linear forward euler; 1-linear exact
    std::vector< double > P11_; // synaptic current evolution parameter
    std::vector< double > P21_; // synaptic current evolution parameter
    std::vector< double > P22_; // synaptic current evolution parameter
    double P30_; // membrane current/voltage evolution parameter
    double P33_; // membrane voltage evolution parameter
    std::vector< double > P31_; // synaptic/membrane current evolution parameter
    std::vector< double > P32_; // synaptic/membrane current evolution parameter

    /** Amplitude of the synaptic current.
              This value is chosen such that a post-synaptic current with
              weight one has an amplitude of 1 pA.
    */
    std::vector< double > PSCInitialValues_; // post synaptic current initial values in pA

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
  static nest::RecordablesMap< glif_lif_psc > recordablesMap_;

};

inline size_t
nest::glif_lif_psc::Parameters_::n_receptors_() const
{
  return tau_syn_.size();
}

inline nest::port
nest::glif_lif_psc::send_test_event( nest::Node& target,
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
nest::glif_lif_psc::handles_test_event( nest::CurrentEvent&,
  nest::port receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c CurrentEvent on port 0. You need to extend the function
  // if you want to differentiate between input ports.
  if ( receptor_type != 0 ){
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return 0;
}

inline nest::port
nest::glif_lif_psc::handles_test_event( nest::DataLoggingRequest& dlr,
  nest::port receptor_type )
{
  // You should usually not change the code in this function.
  // It confirms to the connection management system that we are able
  // to handle @c DataLoggingRequest on port 0.
  // The function also tells the built-in UniversalDataLogger that this node
  // is recorded from and that it thus needs to collect data during simulation.
  if ( receptor_type != 0 ){
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  }
  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
glif_lif_psc::get_status( DictionaryDatum& d ) const
{
  // get our own parameter and state data
  P_.get( d );
  S_.get( d );

  // get information managed by parent class
  Archiving_Node::get_status( d );

  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
glif_lif_psc::set_status( const DictionaryDatum& d )
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
