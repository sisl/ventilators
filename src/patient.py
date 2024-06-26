import json
import hashlib

# Patient related imports
from pulse.cdm.patient import SEPatientConfiguration, SEPatient, eSex
from pulse.cdm.io.patient import serialize_patient_from_file
from pulse.cdm.patient_actions import SEAcuteRespiratoryDistressSyndromeExacerbation
from pulse.cdm.patient_actions import SEDyspnea
from pulse.cdm.patient_actions import SERespiratoryFatigue
from pulse.cdm.patient_actions import SEAsthmaAttack
from pulse.cdm.physiology import eLungCompartment

# Units
from pulse.cdm.scalars import FrequencyUnit, PressureUnit, PressureTimePerVolumeUnit, \
                              TimeUnit, VolumeUnit, VolumePerPressureUnit, VolumePerTimeUnit, \
                              LengthUnit, MassUnit

class Patient:
    def __init__(
        self,
        pulse,
        data_mgr,
        name="patient0",
        sex="Male",
        age=30,
        weight=150,
        height=70,
    ):
        
        # Save the patient params
        self.name = name
        self.sex = sex 
        self.age = float(age)
        self.weight = float(weight)
        self.height = float(height)

        # Load a default patient and set where data can be found
        pc = SEPatientConfiguration()
        # Have to set data root to avoid a seg fault
        data_root_dir = "/pulse/bin/"
        pc.set_data_root_dir(data_root_dir)

        # Create a patient file and use that to initialise the patient
        self.patient_file = self.create_patient_file(
            name=self.name,
            sex=self.sex,
            age=self.age,
            weight=self.weight,
            height=self.height,
        )

        # Set patient via file
        pc.set_patient_file(self.patient_file)

        # Don't need to initialize if we serialize from file
        # https://gitlab.kitware.com/physiology/engine/-/blob/stable/src/python/pulse/howto/HowTo_AsthmaAttack.py#L3
        # Initialize the engine with our configuration
        # In this case we've given the patient a condition (ARDS) so it needs to stabilise to a 
        # steady state (i.e. a converged system) so that we can work with it. Actions however are
        # considred acute responses and do not require stabilization. Conditions can only be
        # changed at run time via exacerbations
        print("Initializing engine")
        if not pulse.initialize_engine(pc, data_mgr):
            print("Unable to stabilize engine on patient initialization")
            return
        # May see something like this: 'Convergence took 26.5s to simulate 437s to get engine to a steady state'
        # But note that the time 'in the sim' is still t=0 until we advance time
        
        # Get default data at time 0s from the engine post initialization
        print("Engine initialized")
        # print("Results at time 0s (the following data modes are being explicitly logged):")
        # pulse.print_results()

    def create_patient_file(
        self,
        name,
        sex,
        age,
        weight,
        height,
        verbose=True,
    ):
        """
        Copy the standard patient file for that sex
        to a temporary location and return the path

        height in inches, weight in lbs, age in years, sex
        a string of either Male or Female

        Simulator doesn't work with geriatrics or pediatrics,
        so the age must be [18, 65]
        """

        # Load the JSON data from the file
        if sex.lower() == "male":
            fp_template = "/mnt/src/patients/StandardMale.json"
        elif sex.lower() == "female":
            fp_template = "/mnt/src/patients/StandardFemale.json"
        else:
            raise Exception(f"{sex} must be either 'Male' or 'Female'")
        with open(fp_template, 'r') as f:
            data = json.load(f)

        # Name it with some hash of the input
        hash_input = hashlib.md5(f"{name}{sex}{age}{weight}{height}".encode()).hexdigest()
        fp_tmp = f"/tmp/{hash_input}.json"

        # Modify the fields
        data["Name"] = name
        data["Sex"] = sex
        data["Age"]["ScalarTime"]["Value"] = age
        data["Weight"]["ScalarMass"]["Value"] = weight
        data["Height"]["ScalarLength"]["Value"] = height

        # Save the modified data to the temporary file
        with open(fp_tmp, 'w') as file:
            json.dump(data, file, indent=4)

        if verbose:
            print(f"Patient file created at {fp_tmp}")
            print(f"\tName: {name}")
            print(f"\tSex: {sex}")
            print(f"\tAge (years): {age}")
            print(f"\tWeight (lb): {weight}")
            print(f"\tHeight (in): {height}")

        return fp_tmp

# ----------------------------------------------------------------
# Diseases to add to patient
# ----------------------------------------------------------------

def ards(
    pulse,
    ards_left=0, 
    ards_right=0,
):
    # Exacerbate ARDS
    exacerbation = SEAcuteRespiratoryDistressSyndromeExacerbation()
    exacerbation.set_comment("Patient's Acute Respiratory Distress Syndrome is altered (exacerbation)")
    exacerbation.get_severity(eLungCompartment.LeftLung).set_value(ards_left)
    exacerbation.get_severity(eLungCompartment.RightLung).set_value(ards_right)
    pulse.process_action(exacerbation)

def dyspnea(
    pulse,
    severity=0,
):
    # Exacerbate dyspnea
    dyspnea = SEDyspnea()
    dyspnea.set_comment("Patient is given dyspnea")
    dyspnea.get_severity().set_value(severity)
    pulse.process_action(dyspnea)