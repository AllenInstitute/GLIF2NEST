#ifndef PIF_LIF_H
#define PIF_LIF_H

#include "namedatum.h"

#include "archiving_node.h"
#include "universal_data_logger.h"

#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "dictdatum.h"



namespace glif
{

  class glif_lif : public nest::Archiving_Node {
  public: 
    glif_lif();
    glif_lif(const glif_lif &n);

    using nest::Node::handle;
    using nest::Node::handles_test_event;

   
    void handle(nest::SpikeEvent&);
    void handle(nest::CurrentEvent&);
    void handle(nest::DataLoggingRequest&);

    nest::port handles_test_event(nest::SpikeEvent&, nest::port);
    nest::port handles_test_event(nest::CurrentEvent&, nest::port);
    nest::port handles_test_event(nest::DataLoggingRequest&, nest::port);

    nest::port send_test_event(nest::Node &target, nest::port receptor_type, nest::synindex, bool);

    void get_status(DictionaryDatum&) const;
    void set_status(const DictionaryDatum&);

  private:
    void init_state(const Node &proto);

    void init_buffers_();

    void calibrate();

    void update(nest::Time const&, const long, const long);

    friend class nest::RecordablesMap<glif_lif>;
    friend class nest::UniversalDataLogger<glif_lif>;

    struct Parameters_
    {
      double C_m_; // Membrane capactiance
      double I_e_; // DC current,
      double G_; // membrance conductance
      double El_; // resting potential
      double V_reset_; // reset potential
      double V_th_; // threshold

      Parameters_();

      void get(DictionaryDatum&) const;
      void set(const DictionaryDatum&);
    };

    struct State_
    {
      double V_m_;

      State_(const Parameters_&);

      void get(DictionaryDatum &d) const;
      void set(const DictionaryDatum&, const Parameters_&);
    };

    struct Buffers_
    {
      Buffers_(glif_lif&);
      Buffers_(const Buffers_&, glif_lif&);
      nest::UniversalDataLogger<glif_lif> logger_;

      nest::RingBuffer spikes_;
      nest::RingBuffer currents_;
    };

    struct Variables_
    {
    };

    Parameters_ P_;
    State_ S_;
    Variables_ V_;
    Buffers_ B_;

    double get_V_m_() const {
      return S_.V_m_;
    }

    static nest::RecordablesMap<glif_lif> recordablesMap_;
  };

  inline nest::port glif_lif::send_test_event(nest::Node &target, 
					      nest::port receptor_type,
					      nest::synindex, bool) {
    nest::SpikeEvent e;
    e.set_sender(*this);
    return target.handles_test_event(e, receptor_type);
  }

  inline nest::port glif_lif::handles_test_event(nest::SpikeEvent& e,
						 nest::port receptor_type) {
    if(receptor_type != 0)
      throw nest::UnknownReceptorType(receptor_type, get_name());
    else
      return 0;
  }

  inline nest::port glif_lif::handles_test_event(nest::CurrentEvent &e,
						 nest::port receptor_type) {
    if(receptor_type != 0)
      throw nest::UnknownReceptorType(receptor_type, get_name());
    else
      return 0;
  }
  
  inline nest::port glif_lif::handles_test_event(nest::DataLoggingRequest &dlr,
						 nest::port receptor_type) {
    if(receptor_type != 0)
      throw nest::UnknownReceptorType(receptor_type, get_name());
    else
      return B_.logger_.connect_logging_device(dlr, recordablesMap_);
  }

  inline void glif_lif::get_status(DictionaryDatum &d) const {
    P_.get(d);
    S_.get(d);

    Archiving_Node::get_status(d);

    (*d)[nest::names::recordables] = recordablesMap_.get_list();
  }

  inline void glif_lif::set_status(const DictionaryDatum &d) {
    Parameters_ ptmp = P_;
    ptmp.set(d);
    State_ stmp = S_;
    stmp.set(d, ptmp);
    
    Archiving_Node::set_status(d);

    P_ = ptmp;
    S_ = stmp;
  }

}


#endif
