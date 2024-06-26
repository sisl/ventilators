from pulse.cdm.engine import SEDataRequest, SEDataRequestManager

# Units
from pulse.cdm.scalars import FrequencyUnit, PressureUnit, PressureTimePerVolumeUnit, \
                              TimeUnit, VolumeUnit, VolumePerPressureUnit, VolumePerTimeUnit, \
                              LengthUnit, MassUnit

def standard_data_requests():
    """
    Note that in the respiratory problem there are two types of request sources
    
    Physiology data ('actual' data from patient)
    This is what's available (respiratory system):
    https://pulse.kitware.com/_c_d_m_tables.html#RespiratorySystemTable

    From ventilator (will overlap with patient)
    This is what's available:
    https://pulse.kitware.com/_c_d_m_tables.html#MechanicalVentilatorTable
    """
        
    return [
        
        # For the respiratory problem we are concerned with the following:
        
        # ----------------------------------------------------------------
        
        # CONTROLS (changes depending on vent mode, this is for pressure control)
        
        # In this mode, the peak airway pressure is constant (inspiratory pressure + PEEP)
        # and tidal volume is allowed to vary.
        # Within this we have:
        # - PC/CMV (cts mandatory ventilation): all patient's breaths being provided by the ventilator,
        #   what you set is what you get. locked.
        # - PC/AC (assist control): patient gets a minimum of what you set, but they can take breaths
        #   if they want triggered by their attempts
        
        # - Fi02 (fraction of inspired oxygen) [0.21, 1]
        # - Inspiratory pressure (Pinsp) [1, 100]
        # - Inspiratory period (Ti) [0.1, 60]
        # - Respiratory rate (RR) [10, 60]
        # - Positive end expiratory pressure (PEEP) [0, 50]
        # - Slope (controls how quickly the airway pressure is reached) [0, 1]
        # - (optional) I:E ratio (proportions of each breath cycle devoted to the inspiratory and expiratory phases)
        
        # Additionally you can optionally control the patient's trigger flow/pressure
        # for getting a breath
        
        # Note that these are actions - they are saved separately
        
        # ----------------------------------------------------------------
        
        # VITALS (overall good stuff to measure from the patient)
        
        # - ECG III: electrical difference between the left arm and the left leg. The leg 
        #   electrode is the one being explored. Lead III observes the heart from a 120° angle. mV
        # - PLETH (pulse oximeter photoplethysmograph): measure of volumetric changes associated 
        #   with pulsatile arterial blood flow
        # - CO2 pressure (how well is CO2 able to flow out of the body). i.e. the
        #   pressure of CO2 dissolved in the (arterial) blood
        # - heart rate / min
        # - mmHg blood pressure: The larger number is the pressure in the arteries as the heart 
        #   pumps out blood during each beat. This is called the systolic blood pressure.
        #   The lower number is the pressure as the heart relaxes before the next beat. This is 
        #   called the diastolic blood pressure.
        # - SpO2 (oxygen saturation): how much oxygen your blood is carrying as a percentage 
        #   of the maximum it could carry
        # - etCO2: level of carbon dioxide released at end of exhaled breath
        # - awRR: airway respiratory rate
        # - T (temperature in degrees celsius)
        
        SEDataRequest.create_ecg_request("Lead3ElectricPotential"),
        SEDataRequest.create_physiology_request("PulseOximetry"),
        SEDataRequest.create_physiology_request("PulmonaryArterialCarbonDioxidePressure"),
        SEDataRequest.create_physiology_request("HeartRate"),
        SEDataRequest.create_physiology_request("DiastolicArterialPressure"),
        SEDataRequest.create_physiology_request("SystolicArterialPressure"),
        SEDataRequest.create_physiology_request("OxygenSaturation"),
        SEDataRequest.create_physiology_request("EndTidalCarbonDioxideFraction"),
        SEDataRequest.create_physiology_request("RespirationRate"),
        SEDataRequest.create_physiology_request("CoreTemperature"),
        # Also adding this as its useful in the reward model
        # Pa02 - arterial oxygen pressure
        SEDataRequest.create_physiology_request("PulmonaryArterialOxygenPressure"),

        # May need to have a phys request to have a mech V request?
        SEDataRequest.create_physiology_request("InspiratoryExpiratoryRatio"),

        
        # ----------------------------------------------------------------

        # VENTILATOR
        
        # Waveforms
        # - Peak airway pressure (Paw) [-1, 20] cmH20
        # - Flow L/min [-30, 30] L/min
        # - Volume mL: maximum volume of air the lungs can hold after a maximum inhalation
        #   (the sum of the four primary lung volumes) [-100, 500] mL
        
        SEDataRequest.create_mechanical_ventilator_request("AirwayPressure", unit=PressureUnit.cmH2O),
        SEDataRequest.create_mechanical_ventilator_request("InspiratoryFlow", unit=VolumePerTimeUnit.L_Per_s),
        SEDataRequest.create_mechanical_ventilator_request("ExpiratoryFlow", unit=VolumePerTimeUnit.L_Per_s),
        SEDataRequest.create_mechanical_ventilator_request("TotalLungVolume", unit=VolumeUnit.L),
        
        # Readings
        # - PIP (highest pressure measured during respiratory cycle). cmH20
        # - Vt (amount of air moving in and out of lung each breath). mL
        # - Minute ventilation (MVe) is the volume of gas inhaled (inhaled minute volume) or
        #   exhaled (exhaled minute volume) from a person's lungs per minute. L/min
        # - MAP / Pmean. cmH20
        # - RR. /min
        # - etcO2. mmHg
        # - Dynamic lung compliance (Cdyn) is a measurement of lung compliance 
        #   and airway resistance. L/cmH20
        # - I:E ratio
        
        SEDataRequest.create_mechanical_ventilator_request("PeakInspiratoryPressure", unit=PressureUnit.cmH2O),
        SEDataRequest.create_mechanical_ventilator_request("TidalVolume", unit=VolumeUnit.mL),
        SEDataRequest.create_mechanical_ventilator_request("TotalPulmonaryVentilation", unit=VolumePerTimeUnit.L_Per_s),
        SEDataRequest.create_mechanical_ventilator_request("MeanAirwayPressure", unit=PressureUnit.cmH2O),
        SEDataRequest.create_mechanical_ventilator_request("RespirationRate", unit=FrequencyUnit.Per_min),
        SEDataRequest.create_mechanical_ventilator_request("EndTidalCarbonDioxideFraction"),
        # During no gas flow
        SEDataRequest.create_mechanical_ventilator_request("StaticPulmonaryCompliance"),
        # During gas flow
        SEDataRequest.create_mechanical_ventilator_request("DynamicPulmonaryCompliance"),
        SEDataRequest.create_mechanical_ventilator_request("InspiratoryExpiratoryRatio"),      

        SEDataRequest.create_mechanical_ventilator_request("PlateauPressure", unit=PressureUnit.cmH2O),
        
        
        # ----------------------------------------------------------------
        
        # These readings should be enough to give a snapshot of the patient's health

        # Get the pH
        SEDataRequest.create_physiology_request("BloodPH"),
        
    ]