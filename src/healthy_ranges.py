import pandas as pd
from tqdm import tqdm
import numpy as np

import utils

def get_high_flow_ventilator_settings(
    patient,
    state,
):
    """
    For a given patient and state, return the (not clinically recommended)
    high flow ventilator settings. Note that this is there just to
    demonstrate that high flow ventilator settings are not protective
    for a patient
    """
    # TODO this is a placeholder, need to implement the actual logic
    pass

def get_healthy_range(
    unit,
    patient_object,
):
    """
    Is the number representing the unit in the healthy range for the patient?

    i.e., the blood pressure for a male 20 year old can be different than for
    a female 60 year old
    """

    unit = unit.lower()
    age = patient_object.age
    sex = patient_object.sex.lower()
    height = patient_object.height # inches

    # NOTE: turns out there really aren't many differences in healthy ranges
    # between men and women
    
    # NOTE can we get a race breakdown also for future work - need to check what the default race
    # is for the patients for these sources
    if unit == "ph":
        # Normal blood pH range is across the board
        return 7.35, 7.45

    elif unit == "spo2":
        # Oxygen saturation (SpO2)
        # NOTE: take into account COPD?
        # Source: https://pdf.sciencedirectassets.com/272456/1-s2.0-S0735675700X01258/1-s2.0-S0735675799900190/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEHEaCXVzLWVhc3QtMSJGMEQCID2qlP9hGiTqyUH9eLMJmqtOCoNg3%2FAfZGc%2B0Wtc3YCIAiB1L8h1v00m1qW3Ou5aza4ilWJCcbwG1TW5%2Bx32Iv%2F0wSqzBQh6EAUaDDA1OTAwMzU0Njg2NSIMrgP1LMg4QUE1amuVKpAFgp99GZPlrod8MdzIUFsle92cY41s45IM13rkHnKtEnBnzrzV6bzsP6QU53nColm9xZ9Yl%2Fq8CLEt7uI%2BfwjUV0xV6%2B3uQbhezvBRVotGthjy6XdiEeTt%2BFlq5IEVPnhI5w2SoOlzOc7bFJHOlrkxbGeCH7D7nouZ%2Fs5V2mygARkseldk6KG2I3%2BNrrkeOQ23bvsXLRNGkGJJ%2Fs%2BfH3BS0XpDHI5Pk6n4dEEf8%2F3FcVWh%2F2wr4In8GAX3iwigmv1QMhMtUb3ZqwImk2bwSFTHTtydtfgi%2BHgCZo93ymN41I1hIHL1gIWpFyqtOu%2BhNwNnx5%2BCKqOp1fG5oxbHHd3qUPazrc%2FSPmyeR5LJ%2FdZCNjcRp%2BVIO%2BznBZVknQHDzIcrnri98S81zNhlS%2FqFaYe08GmTm%2FDwfIV6HcSE4VVA0XRbIrRpXVjR%2FuscSqrUMSrhlqAD2KetSmi1SflD8PjjwNsz8SIdK0JFQ5EHGqZMp0ZRrhSbwO0OtRzQAPD%2FKesv1jtVOF64MPcc7V8GK1jkNj2zcboTIX5moMwnOB5P4ElTBzcS%2BAxwtH06cbfIZbbM3AaeOYHe7mYq54j3rJolsNEfnN97OPeFCrOREYgCKD2IV5Ucafap87z3H20caU3s8A3gAEsRT9ve36JTczI40GYWzirihuzGCcG0DjcypZiqFwfArQwIMWC4Bdunmf%2FHK5E2dmlvl92gVis1Q0rbpJI4PCkn1oaembfViheGUCljdUvJ06%2Bj%2BiO5WbjETvqy4vMPnXMLAuXdoRBoOc91mrxVqmKc281hLNFZBY4g1dw3QunTU7qRXUaZHaMIhLe8ND9KyrL%2Fn%2BECQajFsykBi5QdnLcMeFmywfGULbgrnucwxPivwwY6sgGqAM02%2FEehWmil7o2msKjO%2F%2F3CXMPDNu%2FPo1hfffwibAfY41qUg3B6t6y8xroIssYyEI3zKwpEM2g%2FV64uH74fjhES8XMHmqFOH4c0920XfCfmvO3iZxADxUXv19rnuz1JssqScd%2Fqpyr7%2FDWVbfflWhGNRTHzT8Skq9GyuiX3Tpmuul2JUIpDBtNcMEdSyo5zGPh2Q966wzYTK%2BiD0uiLGNDo6VngI0gYPvjvZ5G99dEp&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250707T173221Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3YMFGHCU%2F20250707%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=fd5d3e1c5f7471ac28fc19f68b9e99526f9066041cfc46280dc56e23480650c8&hash=61b6b2393c99c8ee1dafe78f5770cdfdb760c9f4582984eee45277f698ac50be&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0735675799900190&tid=spdf-c2ba5ec1-d136-4437-bfd4-f39f84edc4b6&sid=a28c87678645594ef709c75-d94f8fa4176dgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0f155f52075d5351005453&rr=95b914a55a22cce4&cc=us&kca=eyJrZXkiOiJuSUM2aGxEUmQ5Z1BhbUh0SEN5cVpkWUsvZU9JWEZVWFVwRTg1Q0JzejlCVHpPbUZ4U2VZdEl3cHBTdkN4OXY3Q0huQll0VVhpaFJvUnhNd1l6QjVRWFE5Sk9lSFBlM05Vamx5NHZKSjNrWVpJSENObG9ZZEFXUTdBNktiNVBEVGpSVjRlMktuTHBON2Jpc0VBcy9FeXY4WWw2WVdoRzdxMzdYWUNuR3p3cDVMVVNWbiIsIml2IjoiMTg4OTY4OTVjM2M4NGQ0NDc5NWE0ZDVkMTcyNTRmMDgifQ==_1751909560158
        return 0.95, 1.00

    elif unit == "pao2":
        # Arterial oxygen pressure (PaO2)
        # These numbers are what Jana said are useful across the board 
        return 75, 100

    elif unit in ("rr", "awrr"):
        # Respiratory rate by age
        # Source: https://emedicine.medscape.com/article/2172054-overview
        if age >= 18:
            return 12, 20
        # NOTE: these ranges are disabled for this study because
        # we are not using them in the population, but these should
        # be uncommented if you want to use them
        # elif age <= 0.25:
        #     return 30, 60  # 0–3 months
        # elif age <= 1:
        #     return 25, 60  # 3–11 months
        # elif age <= 3:
        #     return 20, 40
        # elif age <= 6:
        #     return 20, 40
        elif age <= 12:
            return 14, 30
        else:
            return 12, 20

    elif unit == "hr":
        # Heart rate by age
        # Source: https://emedicine.medscape.com/article/2172054-overview
        if age >= 15:
            return 60, 100
        # NOTE, see comment above
        # elif age <= 0.25:
        #     return 100, 160
        # elif age <= 1:
        #     return 70, 170
        # elif age <= 3:
        #     return 80, 130
        # elif age <= 5:
        #     return 80, 120
        # elif age <= 10:
        #     return 70, 110
        # elif age <= 14:
        #     return 60, 105
        else:
            return 60, 100

    elif unit == "vt":
        # Tidal volume (mL per kg of ideal body weight)
        # Jana provided this formula for ideal body weight
        height_in_inches = height
        if sex == "male":
            ibw_kg = 50 + 2.3 * (height_in_inches - 60)
        else:
            ibw_kg = 45.5 + 2.3 * (height_in_inches - 60)

        lower = 6 * ibw_kg
        upper = 8 * ibw_kg
        return lower, upper

    elif unit == "bps":
        # Systolic blood pressure
        # Sources: 
        # https://www.baptisthealth.com/blog/heart-care/healthy-blood-pressure-by-age-and-gender-chart
        # EMT class at Stanford
        if age < 18:
            return 90, 120  # Pediatric normal range
        elif sex == "male":
            if age < 60:
                return 90, 125
            else:
                return 90, 135
        elif sex == "female":
            if age < 60:
                return 90, 120
            else:
                return 90, 140
        else:
            raise ValueError(f"Sex {sex} not recognized")

    elif unit == "bpd":
        # Diastolic blood pressure
        # As above
        if age < 18:
            return 60, 80
        elif sex == "male":
            if age < 60:
                return 60, 85
            else:
                return 60, 90
        elif sex == "female":
            if age < 60:
                return 60, 80
            else:
                return 60, 85
        else:
            raise ValueError(f"Sex {sex} not recognized")

    elif unit == "ie":
        # Inspiratory:expiratory ratio
        # Jana says this should be here across teh board
        return 0.3, 0.5

    elif unit == "pplat":
        # Plateau pressure
        # This is a lung protective ventilation target, not a physiological 
        # sign, so it doesn't vary by patient typically
        return 0, 30

    elif unit == "temp":
        # Body temperature in °C, NOTE this never deviated 
        # to out of range in all of our simulations
        return 36.5, 37.3

    else:
        raise ValueError(f"Unit {unit} not recognized")