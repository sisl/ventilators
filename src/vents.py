# Mechanical ventilator (incl control schemes) imports
from pulse.cdm.mechanical_ventilator import eSwitch, eDriverWaveform
from pulse.cdm.mechanical_ventilator_actions import SEMechanicalVentilatorConfiguration, \
                                                    SEMechanicalVentilatorContinuousPositiveAirwayPressure, \
                                                    SEMechanicalVentilatorPressureControl, \
                                                    SEMechanicalVentilatorVolumeControl, \
                                                    SEMechanicalVentilatorHold, \
                                                    SEMechanicalVentilatorLeak, \
                                                    eMechanicalVentilator_PressureControlMode, \
                                                    eMechanicalVentilator_VolumeControlMode


# Units
from pulse.cdm.scalars import FrequencyUnit, PressureUnit, PressureTimePerVolumeUnit, \
                              TimeUnit, VolumeUnit, VolumePerPressureUnit, VolumePerTimeUnit, \
                              LengthUnit, MassUnit

class Ventilator:
    def __init__(
        self,
        pulse,
    ):
        # Save the pulse object
        self.pulse = pulse

        # And create an actual ventilator object
        # Ventilator in PC-AC mode
        self.vent = SEMechanicalVentilatorPressureControl()
        self.vent.set_mode(eMechanicalVentilator_PressureControlMode.AssistedControl)

        # Create the hold object
        #self.hold = SEMechanicalVentilatorHold()

        # Note that these objects need to persist and be updated

        # Create an actions dictionary to save the ventilator settings
        # over time
        self.actions = []

        # The first action is 'all zero'
        self.actions.append({
            "Time(s)": 0,
            "fio2": 0,
            "pinsp": 0,
            "ti": 0,
            "rr": 0,
            "peep": 0,
            "slope": 0,
        })

    def update(
        self,
        fio2=0.21,
        pinsp=13,
        ti=1.0,
        rr=12.0,
        peep=5.0,
        slope=0.1,
    ):
        # Construct a dict which will be saved
        action_dict = {}
        # Add all keyword arguments to the dictionary
        for arg_name, arg_value in locals().items():
            if arg_name != 'self' and arg_name != 'action_dict':
                action_dict[arg_name] = arg_value
        action_dict["Time(s)"] = self.pulse.pull_data()[0]

        # Save the action
        self.actions.append(action_dict)

        # Now set the ventilator object and hold settings
        # Fi02 (fraction of inspired oxygen) [0.21, 1]
        self.vent.get_fraction_inspired_oxygen().set_value(fio2)
        # Inspiratory pressure (Pinsp) [1, 100]
        self.vent.get_inspiratory_pressure().set_value(pinsp, PressureUnit.cmH2O)
        # Inspiratory period (Ti) [0.1, 60]
        self.vent.get_inspiratory_period().set_value(ti, TimeUnit.s)
        # Respiratory rate (RR) [10, 60]
        self.vent.get_respiration_rate().set_value(rr, FrequencyUnit.Per_min)
        # Positive end expiratory pressure (PEEP) [0, 50]
        self.vent.get_positive_end_expired_pressure().set_value(peep, PressureUnit.cmH2O)
        # Slope (controls how quickly the airway pressure is reached) [0, 1]
        self.vent.get_slope().set_value(slope, TimeUnit.s)
        # Set the connection
        self.vent.set_connection(eSwitch.On)

        # Apply the ventilator settings
        self.pulse.process_action(self.vent)

    def get_actions(self):
        return self.actions