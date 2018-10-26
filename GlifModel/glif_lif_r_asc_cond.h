#ifndef GLIF_LIF_R_ASC_COND_H
#define GLIF_LIF_R_ASC_COND_H

// Generated includes:
#include "config.h"

#ifdef HAVE_GSL

// C includes:
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

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
 * Author: Binghuang Cai and Kael Dai @ Allen Institute for Brain Science
 *
 */

extern "C" int glif_lif_r_asc_cond_dynamics( double, const double*, double*, void* );

class glif_lif_r_asc_cond : public nest::Archiving_Node
{
public:

  glif_lif_r_asc_cond();

  glif_lif_r_asc_cond( const glif_lif_r_asc_cond& );

  ~glif_lif_r_asc_cond();

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

  // make dynamics function quasi-member
  friend int glif_lif_r_asc_cond_dynamics( double, const double*, double*, void* );

  // The next two classes need to be friends to access the State_ class/member
  friend class nest::RecordablesMap< glif_lif_r_asc_cond >;
  friend class nest::UniversalDataLogger< glif_lif_r_asc_cond >;


  struct Parameters_
  {
	double th_inf_;  	// infinity threshold in mV
    double G_; 			// membrane conductance in nS
    double E_L_; 		// resting potential in mV
    double C_m_; 		// capacitance in pF
    double t_ref_; 		// refractory time in ms

    double a_spike_; 	// threshold additive constant following reset in mV
    double b_spike_;	//spike induced threshold in 1/ms
    double voltage_reset_a_; //voltage fraction following reset coefficient
    double voltage_reset_b_; // voltage additive constant following reset in mV


    std::vector<double> asc_init_; 	// initial values of ASCurrents_ in pA
    std::vector<double> k_; 		// predefined time scale in 1/ms
    std::vector<double> asc_amps_; 	// in pA
    std::vector<double> r_;			// coefficient
    std::vector< double > tau_syn_; // synaptic port time constants in ms
    std::vector< double > E_rev_; // reversal pontiental in mV

    // boolean flag which indicates whether the neuron has connections
    bool has_connections_;

    size_t n_receptors_() const; //!< Returns the size of tau_syn_
    size_t n_ASCurrents_() const; //!< Returns the size of after spike currents

    Parameters_();

    void get( DictionaryDatum& ) const;
    void set( const DictionaryDatum& );
  };


  struct State_
  {
    double V_m_;  // membrane potential in mV
    double ASCurrents_sum_; 	// in pA
    double threshold_; // voltage threshold in mV

    //! Symbolic indices to the elements of the state vector y
    enum StateVecElems
    {
      V_M = 0,
	  ASC,
      DG_SYN,
      G_SYN,
      STATE_VECTOR_MIN_SIZE
    };

    static const size_t NUMBER_OF_FIXED_STATES_ELEMENTS = 1;        // V_M
    static const size_t NUMBER_OF_STATES_ELEMENTS_PER_RECEPTOR = 2; // DG_SYN, G_SYN

    std::vector< double > y_; //!< neuron state

    State_( const Parameters_& );
    State_( const State_& );
    State_& operator=( const State_& );

    void get( DictionaryDatum&, const Parameters_&) const;
    void set( const DictionaryDatum&, const Parameters_& );
  };


  struct Buffers_
  {
    Buffers_( glif_lif_r_asc_cond& );
    Buffers_( const Buffers_&, glif_lif_r_asc_cond& );

    std::vector< nest::RingBuffer > spikes_;   //!< Buffer incoming spikes through delay, as sum
    nest::RingBuffer currents_; //!< Buffer incoming currents through delay,

    //! Logger for all analog data
    nest::UniversalDataLogger< glif_lif_r_asc_cond > logger_;

    /* GSL ODE stuff */
    gsl_odeiv_step* s_;    //!< stepping function
    gsl_odeiv_control* c_; //!< adaptive stepsize control function
    gsl_odeiv_evolve* e_;  //!< evolution function
    gsl_odeiv_system sys_; //!< struct describing system

    // IntergrationStep_ should be reset with the neuron on ResetNetwork,
    // but remain unchanged during calibration. Since it is initialized with
    // step_, and the resolution cannot change after nodes have been created,
    // it is safe to place both here.
    double step_;            //!< step size in ms
    double IntegrationStep_; //!< current integration time step, updated by GSL

    /**
     * Input current injected by CurrentEvent.
     * This variable is used to transport the current applied into the
     * _dynamics function computing the derivative of the state vector.
     * It must be a part of Buffers_, since it is initialized once before
     * the first simulation, but not modified before later Simulate calls.
     */
    double I_stim_;

  };

  struct Variables_
  {
    double t_ref_remaining_; // counter during refractory period, in ms
    double t_ref_total_; // total time of refractory period, in ms

    double last_spike_; // threshold spike component in mV

    /** Amplitude of the synaptic conductance.
        This value is chosen such that an event of weight 1.0 results in a peak conductance of 1 nS
		at t = tau_syn.
    */
    std::vector< double > CondInitialValues_; // synapse conductance intial values in nS
    unsigned int receptor_types_size_;

  };

  //! Read out state vector elements, used by UniversalDataLogger
  template < State_::StateVecElems elem >
  double
  get_y_elem_() const
  {
    return S_.y_[ elem ];
  }

  Parameters_ P_; 
  State_ S_;      
  Variables_ V_;  
  Buffers_ B_;    

  // Mapping of recordables names to access functions
  static nest::RecordablesMap< glif_lif_r_asc_cond > recordablesMap_;
};

inline size_t
allen::glif_lif_r_asc_cond::Parameters_::n_receptors_() const
{
  return tau_syn_.size();
}

inline size_t
allen::glif_lif_r_asc_cond::Parameters_::n_ASCurrents_() const
{
  return k_.size();
}


inline nest::port
allen::glif_lif_r_asc_cond::send_test_event( nest::Node& target,
  nest::port receptor_type,
  nest::synindex,
  bool )
{
  nest::SpikeEvent e;
  e.set_sender( *this );
  return target.handles_test_event( e, receptor_type );
}

inline nest::port
allen::glif_lif_r_asc_cond::handles_test_event( nest::CurrentEvent&,
  nest::port receptor_type )
{
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );
  return 0;
}

inline nest::port
allen::glif_lif_r_asc_cond::handles_test_event( nest::DataLoggingRequest& dlr,
  nest::port receptor_type )
{
  if ( receptor_type != 0 )
    throw nest::UnknownReceptorType( receptor_type, get_name() );

  return B_.logger_.connect_logging_device( dlr, recordablesMap_ );
}

inline void
glif_lif_r_asc_cond::get_status( DictionaryDatum& d ) const
{
  // get our own parameter and state data
  P_.get( d );
  S_.get( d, P_ );

  // get information managed by parent class
  Archiving_Node::get_status( d );

  ( *d )[ nest::names::recordables ] = recordablesMap_.get_list();
}

inline void
glif_lif_r_asc_cond::set_status( const DictionaryDatum& d )
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

#endif // HAVE_GSL
#endif
