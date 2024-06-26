import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import utils

def load_time_cutoff_data(filepath, t_start, t_end, return_previous=False, make_piecewise=False, piecewise_t_end=-1):
    # Load data from CSV file
    df = pd.read_csv(filepath)
    
    # Make piecewise first if needed
    if make_piecewise:
        df = utils.make_actions_df_piecewise(df, piecewise_t_end)

    # Drop everything where the time key is outside of the t_start
    # t_end range
    time_key = 'Time(s)'
    df_cut = df[(df[time_key] >= t_start) & (df[time_key] <= t_end)]
    
    # If nothing was found in this range then return the last 
    # entry even if the time is out of range
    if return_previous and df_cut.empty:
        closest_entry = df[df[time_key] < t_start].iloc[-1:]
        return closest_entry
    
    return df_cut

# A variety of helpers to make things look like a monitor 
# that you might find in a hospital
def _monitor_like_visuals(figure, keep_axes=False):
    # Make everything look nice
    for i, ax in enumerate(figure.axes):
        if not keep_axes:
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.set_xticks([])
            ax.set_yticks([])     
        ax.set_facecolor('black')
    
# This helper function plots a big number in an axis
def monitor_text(ax, plot_text, fontsize=24, color="grey"):
    ax.text(
        0.5, 0.5, 
        plot_text, 
        va="center", ha="center", 
        color=color, 
        fontsize=fontsize, 
    )
    
# This helper function plots titles on plots
def monitor_label(ax, label, fontsize=10, color="grey", bbox=dict(facecolor='black', alpha=0.8), y=0.945):
    ax.text(
        0.015, y, 
        label, 
        transform=ax.transAxes, 
        va='top', ha='left', 
        fontsize=fontsize, 
        color=color,
        bbox=bbox,
    )

def plot_vitals(filepath, t_start=-1, t_end=999999999):
    """
    Plots how the patient's vitals evolved over the course of the time period
    provided
    """
    
    # Start by loading the data at this time period
    df = load_time_cutoff_data(filepath, t_start, t_end)
    
    # ----------------------------------------------------------------
    
    # - 1 column on the left with 3 rows
    # - in each of the 3 rows we have a time series plot
    #   for ecgIII, pleth, and c02 (partial pressure)
    # - 1 column on the right with 6 rows
    # - in each of the 6 rows we have a reading averaged over the time period:
    #   - heart rate (single number, green, units /min)
    #   - blood pressure (systolic, diastolic, red, units mmHg)
    #   - SpO2 (single number, cyan blue, units %) / Pa02
    #   - etCO2 (single number, yellow)
    #   - awRR (single number, yellow)
    #   - temperature (single number, green, degrees C)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    #fig.patch.set_facecolor('black')
    
    # Achieve this with grid specs
    gs = GridSpec(6, 2, width_ratios=[6,1])
    left_axes = [ fig.add_subplot(gs[0+2*i:2+2*i, 0]) for i in range(3) ]
    right_axes = [ fig.add_subplot(gs[i, 1]) for i in range(6) ] 
    all_axes = [*left_axes, *right_axes]
    
    _monitor_like_visuals(fig)
    
    # This is the configuration of each plot in order (all left, all right)
    config = [
        {
            'label': 'ECGIII mV',
            'color': 'limegreen',
            'type': 'waveform',
            'data': df['ECG-Lead3ElectricPotential(mV)'],
        },
        {
            'label': 'PLETH %',
            'color': 'red',
            'type': 'waveform',
            'data': df['PulseOximetry']*100,
        },
        {
            'label': 'PaCO2 mmHg',
            'color': 'yellow',
            'type': 'waveform',
            'data': df['PulmonaryArterialCarbonDioxidePressure(mmHg)'],
        },
        {
            'label': 'hr 1/min',
            'color': 'limegreen',
            'type': 'text',
            'data': round(df['HeartRate(1/min)'].mean()),
        },
        {
            'label': 'bp mmHg',
            'color': 'red',
            'type': 'text',
            'data': f"{round(df['SystolicArterialPressure(mmHg)'].mean())} / {round(df['DiastolicArterialPressure(mmHg)'].mean())}",
            'datafontsize': 14,
        },
        {
            'label': 'SpO2 %',
            'color': 'cyan',
            'type': 'text',
            'data': round(df['OxygenSaturation'].mean()*100),
        },
        {
            'label': 'PaO2 mmHg',
            'color': 'cyan',
            'type': 'text',
            'data': f"{df['PulmonaryArterialOxygenPressure(mmHg)'].mean():.1f}",
        },
        {
            'label': 'etCO2 %',
            'color': 'yellow',
            'type': 'text',
            'data': f"{round(df['EndTidalCarbonDioxideFraction'].mean()*100)}",
            
        },
        {
            'label': 'awRR 1/min',
            'color': 'yellow',
            'type': 'text',
            'data': f"{df['RespirationRate(1/min)'].mean():.1f}",
        },
#         {
#             'label': 'T deg C',
#             'color': 'green',
#         },
    ]
    
    # Plot all data according to configuration
    for i, ax in enumerate(all_axes):
        c = config[i]
        # Slightly different if a waveform
        if c['type'] == "waveform":
            monitor_label(ax, c['label'], color=c['color'], fontsize=12)
            ax.plot(c['data'], color=c['color'])
            
        else:
            if 'datafontsize' in c.keys():
                fontsize = c['datafontsize']
            else:
                fontsize = 24
            monitor_label(ax, c['label'], fontsize=8, color=c['color'], bbox=None, y=0.975)
            monitor_text(ax, c['data'], fontsize=fontsize, color=c['color'])
    
    fig.tight_layout()
    
    return fig

def plot_ventilator(fp_states, fp_actions, t_start=-1, t_end=999999999):
    """
    Plots how the ventilator readings changed over the course of the time period
    provided
    (this are only 'on' when the vent is 'on')
    """
    
    # Start by loading the data at this time period
    df_states  = load_time_cutoff_data(fp_states, t_start, t_end)
    df_actions = load_time_cutoff_data(
        fp_actions, 
        t_start, t_end, 
        return_previous=True, 
        make_piecewise=True, 
        piecewise_t_end=load_time_cutoff_data(fp_states, t_start=-1, t_end=999999999)['Time(s)'].max()
    )

    # ----------------------------------------------------------------

    # - split the screen into two rows.
    # - the top row is split into 2 columns
    # - in the left column we have 3 rows, where 
    #   in each row is a timeseries plot. The three
    #   quantities plotted are:
    #   - airway pressure
    #   - flow (inspiratory and expiratory plotted on top of eachother)
    #   - total lung volume
    # - the right column we split into a grid of 4 rows and 2 columns
    #   and include numbers averaged over the time period provided
    #   - PIP
    #   - Vt
    #   - MVe
    #   - MAP
    #   - RR
    #   - etCO2
    #   - 2 numbers here to describe static and dynamic pulmonary compliance
    #   - I/E ratio
    # - the outer level bottom row is split into a grid of 7
    #   readings which are the control inputs to the ventilator:
    #   - FiO2
    #   - P_insp
    #   - Ti
    #   - RR
    #   - PEEP
    #   - slope
    #   - hold
    
    fig = plt.figure(figsize=(8, 5), dpi=300)
    
    # Achieve this with grid specs
    num_control_inputs = 6
    gs = GridSpec(4, num_control_inputs + 2)
    control_axes  = [ fig.add_subplot(gs[3, i]) for i in range(num_control_inputs) ]
    waveform_axes = [ fig.add_subplot(gs[i, 0:num_control_inputs]) for i in range(3) ]
    text_axes     = [ fig.add_subplot(gs[i//2, num_control_inputs+i%2]) for i in range(8) ]
    all_axes      = [*control_axes, *waveform_axes, *text_axes]
    
    _monitor_like_visuals(fig)
       
    config = [
        # Start with the controls
        {
            'label': 'FiO2 Ϙ',
            'color': 'white',
            'type': 'control',
            'data': f"{df_actions['fio2'].mean():.2f}",
        },
        {
            'label': 'Pinsp. \ncmH20',
            'color': 'white',
            'type': 'control',
            'data': f"{round(df_actions['pinsp'].mean())}",
        },
        {
            'label': 'Ti s',
            'color': 'white',
            'type': 'control',
            'data': f"{df_actions['ti'].mean():.1f}",
        },
        {
            'label': 'RR \n1/min',
            'color': 'white',
            'type': 'control',
            'data': f"{df_actions['rr'].mean():.1f}",
        },
        {
            'label': 'PEEP \ncmH20',
            'color': 'white',
            'type': 'control',
            'data': f"{df_actions['peep'].mean():.1f}",
        },
        {
            'label': 'Slope',
            'color': 'white',
            'type': 'control',
            'data': f"{df_actions['slope'].mean():.2f}",
        },
        # TODO: should we add an explicit hold on/off? should be clear 
        # from plots if this thing is on?
        # Now the waveforms
        {
            'label': 'Paw cmH20',
            'color': 'red',
            'type': 'waveform',
            'data': df_states['MechanicalVentilator-AirwayPressure(cmH2O)'],
        },
        {
            'label': 'Flow L/s',
            'color': 'chartreuse',
            'type': 'waveform',
            'data': df_states['MechanicalVentilator-InspiratoryFlow(L/s)'],
#             'data': [
#                 df_states['MechanicalVentilator-InspiratoryFlow(L/s)'],
#                 df_states['MechanicalVentilator-ExpiratoryFlow(L/s)'],
#             ],
        },
        {
            'label': 'Volume L',
            'color': 'yellow',
            'type': 'waveform',
            'data': df_states['MechanicalVentilator-TotalLungVolume(L)'],
        },
        # Text (numbers/stats)
        {
            'label': 'PIP cmH20',
            'color': 'cyan',
            'type': 'text',
            'data': round(df_states['MechanicalVentilator-PeakInspiratoryPressure(cmH2O)'].mean()),
        },
        {
            'label': 'Vt L',
            'color': 'cyan',
            'type': 'text',
            'data': round(df_states['MechanicalVentilator-TidalVolume(mL)'].mean()),
        },
        {
            'label': 'MVe L/min',
            'color': 'limegreen',
            'type': 'text',
            'data': f"{df_states['MechanicalVentilator-TotalPulmonaryVentilation(L/s)'].mean():.2f}",
        },
        {
            'label': 'MAP cmH20',
            'color': 'limegreen',
            'type': 'text',
            'data': f"{df_states['MechanicalVentilator-MeanAirwayPressure(cmH2O)'].mean():.2f}",
        },
        {
            'label': 'RR 1/min',
            'color': 'red',
            'type': 'text',
            'data': f"{df_states['MechanicalVentilator-RespirationRate(1/min)'].mean():.1f}",
        },
        {
            'label': 'etCO2 %',
            'color': 'magenta',
            'type': 'text',
            'data': f"{df_states['MechanicalVentilator-EndTidalCarbonDioxideFraction'].mean()*100:.2f}",
        },
        {
            'label': 'Cdyn \nL/cmH20',
            'color': 'yellow',
            'type': 'text',
            'data': f"{df_states['MechanicalVentilator-DynamicPulmonaryCompliance(L/cmH2O)'].mean():.2f}",
        },
        {
            'label': 'I:E',
            'color': 'magenta',
            'type': 'text',
            'data': f"{df_states['MechanicalVentilator-InspiratoryExpiratoryRatio'].mean():.2f}",
        },
    ]
    
    # Plot all data according to configuration
    for i, ax in enumerate(all_axes):
        c = config[i]
        
        # Slightly different if a waveform
        if c['type'] == "waveform":
            monitor_label(ax, c['label'], color=c['color'], fontsize=12)
            # If the data is a list then plot each
            if isinstance(c['data'], list):
                # Tacky but there's only one case of this and that's
                # unlikely to change
                ax.plot(c['data'][0], color=c['color'], label="insp")
                ax.plot(c['data'][0], color="white", label="exp")
                ax.legend()
            else:
                ax.plot(c['data'], color=c['color'])
        
        else:
            # For both text and controls
            if 'datafontsize' in c.keys():
                fontsize = c['datafontsize']
            else:
                fontsize = 18
            monitor_label(ax, c['label'], fontsize=8, color=c['color'], bbox=None, y=0.975)
            monitor_text(ax, c['data'], fontsize=fontsize, color=c['color'])
    
    fig.tight_layout()
    
    return fig

def plot_simulation(fp_states, fp_actions):
    # - plot two columns
    # - the left column will show a time series of all input variables over the whole simulation
    # - plot reward on the left column too
    # - on the right are vitals/blood gases/etc.
    # - so it will just be entirely timeseries' all up
    
    # Start by loading the data at this time period
    df_states  = load_time_cutoff_data(fp_states, t_start=-1, t_end=999999999)
    df_actions = load_time_cutoff_data(
        fp_actions, 
        t_start=-1, t_end=999999999, 
        return_previous=True, 
        make_piecewise=True,
        piecewise_t_end=load_time_cutoff_data(fp_states, t_start=-1, t_end=999999999)['Time(s)'].max()
    )
    
    fig = plt.figure(figsize=(8, 9), dpi=300)
    
    # Achieve this with grid specs
    num_control_inputs = 6
    num_readings = 9
    gs = GridSpec(num_readings, 2)
    control_axes  = [ fig.add_subplot(gs[i, 0]) for i in range(num_control_inputs) ]
    # TODO add reward axes
    readings_axes = [ fig.add_subplot(gs[i, 1]) for i in range(num_readings) ]
    all_axes      = [*control_axes, *readings_axes]
    
    _monitor_like_visuals(fig, keep_axes=True)
    
    config = [
        # Start with the controls
        {
            'label': 'FiO2 Ϙ',
            'color': 'white',
            'type': 'waveform',
            'times': df_actions["Time(s)"],
            'data': df_actions['fio2'],
        },
        {
            'label': 'Pinsp. cmH20',
            'color': 'white',
            'type': 'waveform',
            'times': df_actions["Time(s)"],
            'data': df_actions['pinsp'],
        },
        {
            'label': 'Ti s',
            'color': 'white',
            'type': 'waveform',
            'times': df_actions["Time(s)"],
            'data': df_actions['ti'],
        },
        {
            'label': 'RR 1/min',
            'color': 'white',
            'type': 'waveform',
            'times': df_actions["Time(s)"],
            'data': df_actions['rr'],
        },
        {
            'label': 'PEEP cmH20',
            'color': 'white',
            'type': 'waveform',
            'times': df_actions["Time(s)"],
            'data': df_actions['peep'],
        },
        {
            'label': 'Slope s',
            'color': 'white',
            'type': 'waveform',
            'times': df_actions["Time(s)"],
            'data': df_actions['slope'],
        },
        # Now the readings of interest
        {
            'label': 'hr 1/min',
            'color': 'limegreen',
            'type': 'waveform',
            'times': df_states["Time(s)"],
            'data': df_states['HeartRate(1/min)'],
        },
        {
            'label': 'bp mmHg',
            'color': 'red',
            'type': 'waveform',
            'times': df_states["Time(s)"],
            'data': [
                df_states['SystolicArterialPressure(mmHg)'],
                df_states['DiastolicArterialPressure(mmHg)'],
            ],
        },
        {
            'label': 'PLETH %',
            'color': 'red',
            'type': 'waveform',
            'times': df_states["Time(s)"],
            'data': df_states['PulseOximetry']*100,
        },
        {
            'label': 'awRR 1/min',
            'color': 'yellow',
            'type': 'waveform',
            'times': df_states["Time(s)"],
            'data': df_states['RespirationRate(1/min)'],
        },
        {
            'label': 'SpO2 %',
            'color': 'cyan',
            'type': 'waveform',
            'times': df_states["Time(s)"],
            'data': df_states['OxygenSaturation']*100,
        },
        {
            'label': 'PaO2 mmHg',
            'color': 'cyan',
            'type': 'waveform',
            'times': df_states["Time(s)"],
            'data': df_states['PulmonaryArterialOxygenPressure(mmHg)'],
        },
        {
            'label': 'etCO2 %',
            'color': 'yellow',
            'type': 'waveform',
            'times': df_states["Time(s)"],
            'data': df_states['EndTidalCarbonDioxideFraction']*100,
            
        },
        {
            'label': 'PaCO2 mmHg',
            'color': 'yellow',
            'type': 'waveform',
            'times': df_states["Time(s)"],
            'data': df_states['PulmonaryArterialCarbonDioxidePressure(mmHg)'],
        },
        # From the ventilator
        {
            'label': 'I:E (vent)',
            'color': 'magenta',
            'type': 'waveform',
            'times': df_states["Time(s)"],
            'data': df_states['MechanicalVentilator-InspiratoryExpiratoryRatio'],
        },
        
    ]
    
    # Plot all data according to configuration
    for i, ax in enumerate(all_axes):
        c = config[i]
        
        # Slightly different if a waveform
        if c['type'] == "waveform":
            monitor_label(ax, c['label'], color=c['color'], fontsize=9, y=0.925)
            
            if isinstance(c['data'], list):
                # If the data is a list then plot each

                # Tacky but there's only one case of this and that's
                # unlikely to change
                ax.plot(c['times'], c['data'][0], color=c['color'])
                ax.plot(c['times'], c['data'][1], color=c['color'], linestyle="--")

                mm = [
                    round(min(c['data'][1]),3), 
                    round(max(c['data'][0]),3),
                ]
                ax.set_yticks(mm)
                ax.set_yticklabels(mm)
            else:
                # Otherwise just plot the data as is
                ax.plot(c['times'], c['data'], color=c['color'])

                if min(c['data']) != max(c['data']):
                    mm = [
                        round(min(c['data']),3), 
                        round(max(c['data']),3),
                    ]
                    ax.set_yticks(mm)
                    ax.set_yticklabels(mm)

                    #print(c['label'], mm)
                    
            ax.set_xticks([])
        
        else:
            # For both text and controls
            if 'datafontsize' in c.keys():
                fontsize = c['datafontsize']
            else:
                fontsize = 18
            monitor_label(ax, c['label'], fontsize=8, color=c['color'], bbox=None, y=0.90)
            monitor_text(ax, c['data'], fontsize=fontsize, color=c['color'])
    
    fig.tight_layout()
    
    return fig