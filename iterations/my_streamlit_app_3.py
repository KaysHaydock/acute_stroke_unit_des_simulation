from distribution import Exponential, Lognormal
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import itertools
import simpy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import streamlit as st
sns.set()

sns.set_style('whitegrid')
sns.set_palette("pastel")

# =============================================================================
# Global Parameters and Model Settings
# =============================================================================

# RUN_LENGTH
WARM_UP_PERIOD = 365*3

DEFAULT_RESULTS_COLLECTION_PERIOD = 365*10

RUN_LENGTH = DEFAULT_RESULTS_COLLECTION_PERIOD + WARM_UP_PERIOD

# default number of repetitions
DEFAULT_N_REPS = 5

# default mean inter-arrival times(exp)
IAT_STROKE = 1.2
IAT_TIA = 9.3
IAT_COMPLEX_NEURO = 3.6
IAT_OTHER = 3.2

# Default Length of Stay (LOS) parameters
# (mean, stdev for Lognormal distribution
LOS_STROKE = (7.4, 8.6)
LOS_TIA = (1.8, 2.3)
LOS_COMPLEX_NEURO = (4.0, 5.0)
LOS_OTHER = (3.8, 5.2)
LOS_STROKE_NESD = (7.4, 8.6)
LOS_STROKE_ESD = (4.6, 4.8)
LOS_STROKE_MORTALITY = (7.0, 8.7)

# Acute Stroke Unit (ASU) bed capacity
ASU_BEDS = 10

# % where patients go after the Acute Stroke Unit (ASU)
TRANSFER_PROBABILITIES = {
    "stroke": {"rehab": 0.24, "esd": 0.13, "other": 0.63},
    "tia": {"rehab": 0.01, "esd": 0.01, "other": 0.98},
    "complex_neuro": {"rehab": 0.11, "esd": 0.05, "other": 0.84},
    "other": {"rehab": 0.05, "esd": 0.10, "other": 0.85}}

# sampling settings, 4 for arrivals, 4 for LOS
# and for 4 transfer probabilities
N_STREAMS = 12
DEFAULT_RND_SET = 101

# Boolean switch to simulation results as the model runs
TRACE = False

def trace(msg):
    if TRACE:
        st.write(msg)

# =============================================================================
# Simulation Classes and Functions
# =============================================================================

class Auditor:
    """
    Records and analyzes the performance
    metrics of the Acute Stroke Unit model.
    """

    def __init__(self):
        """
        Initialises the Auditor with empty lists for delay times and bed occupancy.

        Attributes
        ----------
        delay_time : list
            Stores tuples of (patient_type, delay_occurred, timestamp).
        bed_occupancy : list
            Stores tuples of (occupancy, timestamp).
        """
        self.delay_time = []  # Stores (patient_type, delay_occurred, timestamp)
        self.bed_occupancy = []  # Stores (occupancy, timestamp)

    def record_delay(self, patient_type, delay, timestamp):
        """
        Records the delay information for a patient.

        Parameters
        ----------
        patient_type : str
            The type of patient (e.g., 'stroke', 'TIA').
        delay : float
            The delay time experienced by the patient.
        timestamp : float
            The simulation time at which the delay occurred.
        """
        self.delay_time.append((patient_type, int(delay > 0), timestamp))

    def record_occupancy(self, occupancy, timestamp):
        """
        Records the bed occupancy at a given time.

        Parameters
        ----------
        occupancy : int
            The number of beds occupied.
        timestamp : float
            The simulation time at which the occupancy is recorded.
        """
        self.bed_occupancy.append((occupancy, timestamp))

    def compute_delay_prob(self, warmup=0):
        """
        Computes the delay probability after the warm-up period.

        Parameters
        ----------
        warmup : int, optional
            The warm-up period duration to exclude from
            the analysis (default is 0).

        Returns
        -------
        dict
            A dictionary with overall delay probability
            and delay probability by patient type.
        """
        df = pd.DataFrame(self.delay_time, columns=["type", "delay", "time"])
        df = df[df["time"] >= warmup]
        return {
            "overall": df["delay"].mean() if not df.empty else 0.0,
            "type": df.groupby("type")["delay"].mean().to_dict(),
        }

    def compute_bed_utilization(self, warmup=0):
        """
        Computes the average bed utilization after
        the warm-up period.

        Parameters
        ----------
        warmup : int, optional
            The warm-up period duration to exclude
            from the analysis (default is 0).

        Returns
        -------
        dict
            A dictionary with the overall average bed utilization.
        """
        df = pd.DataFrame(self.bed_occupancy, columns=["occupancy", "time"])
        df = df[df["time"] >= warmup]
        return {"overall": df["occupancy"].mean() if not df.empty else 0.0}


class Experiment:
    """
    Encapsulates the concept of an experiment for the Acute Stroke Unit simulation.
    Manages parameters, PRNG streams, and results.
    """

    def __init__(
        self,
        random_number_set=DEFAULT_RND_SET,
        n_streams=N_STREAMS,
        iat_stroke=IAT_STROKE,
        iat_tia=IAT_TIA,
        iat_complex_neuro=IAT_COMPLEX_NEURO,
        iat_other=IAT_OTHER,
        asu_beds=ASU_BEDS,
        los_stroke=LOS_STROKE,
        los_tia=LOS_TIA,
        los_complex_neuro=LOS_COMPLEX_NEURO,
        los_other=LOS_OTHER,
        transfer_probabilities=TRANSFER_PROBABILITIES,

    ):
        """
        Initialise default parameters.
        """
        # Sampling settings
        self.random_number_set = random_number_set
        self.n_streams = n_streams

        # Model parameters
        self.iat_stroke = iat_stroke
        self.iat_tia = iat_tia
        self.iat_complex_neuro = iat_complex_neuro
        self.iat_other = iat_other
        self.asu_beds = asu_beds

        # LOS Parameters
        self.los_stroke = los_stroke
        self.los_tia = los_tia
        self.los_complex_neuro = los_complex_neuro
        self.los_other = los_other

        # Transfer probabilities
        self.transfer_probabilities = transfer_probabilities

        # Initialise results storage
        self.init_results_variables()

        # Initialise sampling distributions (RNGs)
        self.init_sampling()

        # Audit object
        self.auditor = Auditor()

    def set_random_no_set(self, random_number_set):
        """
        Controls the random sampling by re-seeding.
        """
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_sampling(self):
        """
        Creates the distributions used by the model and initialises
        the random seeds of each.
        """
        # Create a new seed sequence
        seed_sequence = np.random.SeedSequence(self.random_number_set)
        # Produce n non-overlapping streams
        self.seeds = seed_sequence.spawn(self.n_streams)

        # Prepare a list of RNGs
        rng_list = [np.random.default_rng(s.entropy) for s in self.seeds]

        # Inter-arrival time distributions
        self.arrival_stroke = Exponential(self.iat_stroke, self.seeds[0])
        self.arrival_tia = Exponential(self.iat_tia, self.seeds[1])
        self.arrival_complex_neuro = Exponential(self.iat_complex_neuro, self.seeds[2])
        self.arrival_other = Exponential(self.iat_other, self.seeds[3])

        # LOS distributions using stored parameters
        self.los_distributions = {
            "stroke": Lognormal(self.los_stroke[0], self.los_stroke[1], self.seeds[4]),
            "tia": Lognormal(self.los_tia[0], self.los_tia[1], self.seeds[5]),
            "complex_neuro": Lognormal(self.los_complex_neuro[0], self.los_complex_neuro[1], self.seeds[6]),
            "other": Lognormal(self.los_other[0], self.los_other[1], self.seeds[7]),
        }

        # RNGs specifically for transfer choices (1 per patient type)
        self.transfer_rngs = {
            "stroke": rng_list[8],
            "tia": rng_list[9],
            "complex_neuro": rng_list[10],
            "other": rng_list[11],
        }

    def init_results_variables(self):
        """
        Initialises all the experiment variables used in results collection.
        """
        self.results = {
            "n_stroke": 0,
            "n_tia": 0,
            "n_complex_neuro": 0,
            "n_other": 0,
            "n_patients": 0,
            "stroke_transfer": {"rehab": 0, "esd": 0, "other": 0},
            "tia_transfer": {"rehab": 0, "esd": 0, "other": 0},
            "complex_neuro_transfer": {"rehab": 0, "esd": 0, "other": 0},
            "other_transfer": {"rehab": 0, "esd": 0, "other": 0},
            "total_transfers": {"rehab": 0, "esd": 0, "other": 0},
        }
        
class Patient:
    """
    Represents a patient in the Acute Stroke Unit simulation.

    Attributes
    ----------
    patient_id : int
        Unique identifier for the patient.
    env : simpy.Environment
        The simulation environment.
    args : Experiment
        Experiment object containing simulation parameters and random streams.
    acute_stroke_unit : AcuteStrokeUnit
        Reference to the ASU resource (bed capacity, queue).
    patient_type : str
        Type of patient ('stroke', 'TIA', 'complex_neuro', 'other').
    waiting_time : float
        Time patient waits for a bed.
    """
    def __init__(self, patient_id, env, args, acute_stroke_unit, patient_type):
        self.patient_id = patient_id
        self.env = env
        self.args = args
        self.acute_stroke_unit = acute_stroke_unit
        self.patient_type = patient_type
        self.waiting_time = 0.0

    def treatment(self):
        """
        Simulates the patients treatment process:
        - Requests a bed (may wait if unavailable).
        - Records waiting time and occupancy.
        - Simulates length of stay (LOS).
        - Upon discharge, calls `transfer()` to determine next destination.

        Parameters
        ----------
        audit : Audit
            Audit object used to record metrics such as waiting time and bed occupancy.
        """
        arrival_time = self.env.now
        los_distribution = self.args.los_distributions[self.patient_type]

        # Arrival message
        trace(f"Patient {self.patient_id} ({self.patient_type.upper()}) arrives at {arrival_time:.2f}.")

        # Request bed from the ASU
        with self.acute_stroke_unit.beds.request() as request:
            yield request
            self.delay= self.env.now - arrival_time
            self.args.auditor.record_delay(self.patient_type, self.delay, self.env.now)
            los = los_distribution.sample()

            # Bed assigned message
            trace(f"Patient {self.patient_id} ({self.patient_type.upper()}) gets a bed at {self.env.now:.2f}."
                  f" (Waited {self.delay:.2f} days)")

            if self.delay > 0:
                trace(f"Patient {self.patient_id} ({self.patient_type.upper()})  had a delay.)")


            # Simulate length of stay
            yield self.env.timeout(los)

            # Leaving message
            trace(f"Patient {self.patient_id} ({self.patient_type.upper()}) leaves at {self.env.now:.2f}."
                  f" (LOS {los:.2f} days)")


        # Transfer the patient after discharge
        self.transfer()

    def transfer(self):
        """
        Determines the patient's next destination based on the transfer probabilities.
        Logs and updates the experiment's results.
        """
        # Access the RNG and probabilities for this patient type
        rng = self.args.transfer_rngs[self.patient_type]
        p_dict = self.args.transfer_probabilities[self.patient_type]
        destinations = list(p_dict.keys())
        probs = list(p_dict.values())

        # Random draw for the transfer destination
        destination = rng.choice(destinations, p=probs)
        trace(f"Patient {self.patient_id} ({self.patient_type.upper()}) is transferred to {destination.upper()}.")

        # Update results
        self.args.results["total_transfers"][destination] += 1
        self.args.results[f"{self.patient_type}_transfer"][destination] += 1

class AcuteStrokeUnit:
    """
    Models the Acute Stroke Unit (ASU) in the hospital.
    Manages bed resources and handles patient arrivals and admissions.

    Attributes
    ----------
    env : simpy.Environment
        The simulation environment.
    args : Experiment
        Experiment object containing model parameters and random streams.
    beds : simpy.Resource
        SimPy resource representing available ASU beds.
    """
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.beds = simpy.Resource(env, capacity=args.asu_beds)

    def patient_arrivals(self, patient_type, arrival_distribution):
        """
        Generator process that simulates patient arrivals of a specific type.
        For each arrival:
        - Waits based on inter-arrival time (Exponential distribution).
        - Creates a new Patient instance.
        - Starts the treatment process for the patient.
        - Records arrival statistics in Experiment results.

        Parameters
        ----------
        patient_type : str
            Type of patient ('stroke', 'TIA', 'complex_neuro', 'other').
        arrival_distribution : Exponential
            Distribution object for sampling inter-arrival times.
        audit : Audit
            Audit object for recording delays and occupancy.
        """
        for patient_count in itertools.count(start=1):
            inter_arrival_time = arrival_distribution.sample()
            yield self.env.timeout(inter_arrival_time)

            # Track patient count
            self.args.results[f"n_{patient_type}"] += 1
            self.args.results["n_patients"] += 1

            trace(f"{self.env.now:.2f}: {patient_type.upper()} arrival.")

            new_patient = Patient(patient_count, self.env, self.args, self, patient_type)
            self.env.process(new_patient.treatment())

    def track_occupancy(self):
        """
        Track and record bed occupancy daily.
        """
        while True:
            occ = self.beds.count / self.beds.capacity
            self.args.auditor.record_occupancy(occ, self.env.now)
            yield self.env.timeout(1)


def single_run(experiment,
               rep=0,
               run_length=RUN_LENGTH,
               warm_up=0):

    """
    Perform a single run of the model and return the results.

    Parameters
    ----------
    experiment : Experiment
        The experiment/parameters to use with model
    rep : int
        The replication number (used to set random seeds).
    run_length : float
        The run length of the model in days (default = the constant set).
    """
    # 1. Reset results for each run
    experiment.init_results_variables()

    # 2. Set the random number set for this run
    experiment.set_random_no_set(rep)

    # 3. Create a fresh environment and an AcuteStrokeUnit
    env = simpy.Environment()
    auditor = Auditor()
    asu = AcuteStrokeUnit(env, experiment)


   # 4. Create patient arrival processes for different types of patients
    env.process(asu.patient_arrivals("stroke", experiment.arrival_stroke))
    env.process(asu.patient_arrivals("tia", experiment.arrival_tia))
    env.process(asu.patient_arrivals("complex_neuro", experiment.arrival_complex_neuro))
    env.process(asu.patient_arrivals("other", experiment.arrival_other))
    trace(f"Rep {rep}: Patient arrival processes started.")


    # track the bed occupancy daily
    env.process(asu.track_occupancy())

    # 5. Run the simulation
    env.run(until=run_length + warm_up)
    trace(f"Rep {rep}: Simulation run completed.")

    # 6. Trace summary of total patients
    total_patients = sum(experiment.results[key] for key in experiment.results if key.startswith("n_"))
    trace(f"Final summary for rep={rep}: {total_patients} total patients.")

    # 7. Return experiment.results
    return experiment



def multiple_replications(scenario,
                          rc_period=DEFAULT_RESULTS_COLLECTION_PERIOD,
                          n_reps=DEFAULT_N_REPS,
                          warmup=0):
    '''
    Perform multiple replications of the model.

    Params:
    ------
    scenario: Scenario
        Parameters/arguments to configure the model

    rc_period: float, optional (default=DEFAULT_RESULTS_COLLECTION_PERIOD)
        results collection period.
        the number of minutes to run the model beyond warm up
        to collect results

    warm_up: float, optional (default=0)
        initial transient period.  no results are collected in this period

    n_reps: int, optional (default=DEFAULT_N_REPS)
        Number of independent replications to run.

    n_jobs, int, optional (default=-1)
        No. replications to run in parallel.

    Returns:
    --------
    List
    '''
   # Create a list to store the results of each replication
    rep_results = []
    overall = []
    reps = []
    bed_utilisation = []
    for rep in range(n_reps):
        # Create a new experiment with the current bed capacity
        exp_temp =scenario
        # Run the simulation using the default run length and current replication number
        res = single_run(exp_temp, rep=rep, run_length=rc_period)
        rep_results.append(res.results["n_patients"])
        ovell_p = res.auditor.compute_delay_prob(warmup=warmup)
        util = res.auditor.compute_bed_utilization(warmup=warmup)
        overall.append(ovell_p["overall"])
        bed_utilisation.append(util["overall"])
        info = f"Rep {rep +1 } of {n_reps}"
        reps.append(info)


    # format and return results in a dataframe

    df =pd.DataFrame()
    df["Patients"] = rep_results
    df["Overall Delay Probability"] = overall
    df[f"Reps:{n_reps}"] = reps
    df["Bed Utilisation"] = bed_utilisation
    df = df.set_index(f"Reps:{n_reps}")
    return df



def multiple_runs(scenario, rep, beds_start, beds_end, warmup):
    """
    Perform multiple runs of the simulation across a range of bed capacities.

    For each bed capacity (from beds_start to beds_end), runs the simulation
    'rep' times and calculates average patient count and delay probability.

    Parameters
    ----------
    rep : int
        Number of replications for each bed capacity.
    beds_start : int
        Starting number of beds to simulate.
    beds_end : int
        Ending number of beds to simulate (inclusive).

    Returns
    -------
    summary_df : pandas.DataFrame
        A summary DataFrame containing:
            - Bed Capacity
            - Average number of patients processed
            - Overall delay probability
            - '1 in every n patients delayed' metric (reciprocal of delay probability)
    """
    def run_single_experiment(bed, rep):
        experiment = scenario
        experiment.asu_beds = bed
        result =single_run(experiment, rep=rep)
        delay_p = experiment.auditor.compute_delay_prob(warmup=warmup)
        util = experiment.auditor.compute_bed_utilization(warmup=warmup)

        return result["n_patients"], delay_p["overall"],util["overall"]


    summary_list = []

    for bed in range(beds_start, beds_end+1):
        print(f"Running experiment with {bed} beds and {rep} repetitions, end={beds_end}")
        results = Parallel(n_jobs=-1)(delayed(run_single_experiment)(bed, rep) for rep in range(rep))
        rep_results, delay_p, bed_utill = zip(*results)

        overall_avg = np.mean(delay_p)
        avg_patients = np.mean(rep_results)
        overall_avg_util = np.mean(bed_utill)
        summary_list.append({"Bed Capacity": bed,
                             "Avg. Patients": avg_patients,
                             "Overall Delay Probability": overall_avg,
                             "1 in very n Patients":
                                 1/overall_avg
                                 if overall_avg != 0 else float('inf')})

    print("Finished running all experiments")
    summary_df = pd.DataFrame(summary_list)
    return summary_df


def warmup_analysis(run_length=RUN_LENGTH, reps=5, interval=50):
    """
    Warm-up period analysis.

    Parameters
    ----------
    run_length : int
        Total run length of the simulation.
    reps : int
        Number of replications.
    interval : int
        Interval for warm-up period analysis.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the warm-up analysis results.
    """
    results = []

    for i in range(1, run_length, interval):
        experiment = Experiment()
        multi_rep = multiple_replications(scenario=experiment,
                                          rc_period=i,
                                          n_reps=reps)

        delay_p = multi_rep["Overall Delay Probability"].mean()
        util = multi_rep["Bed Utilisation"].mean()

        results.append({"Run Length": i, "reps": reps, "Delay Probability": delay_p, "Bed Utilisation": util})

    df = pd.DataFrame(results)
    return df

# =============================================================================
# 3 Scenarios
# =============================================================================

# Scenario 0: Current Admissions
experiment_default = Experiment()
results_default = multiple_runs(scenario=experiment_default,
                                beds_start=9,
                                beds_end=17,
                                warmup=WARM_UP_PERIOD,
                                rep=150)
results_default['Scenario'] = 'Current Admissions'

results_default = results_default.round({
    "Overall Delay Probability": 2,
})

results_default["1 in very n Patients"] = results_default["1 in very n Patients"].round(0).astype(int)
results_default

# Scenario 1: 5% Increase in Admissions
experiment_increase = Experiment(
    iat_stroke=IAT_STROKE * 1.05,
    iat_tia=IAT_TIA * 1.05,
    iat_complex_neuro=IAT_COMPLEX_NEURO * 1.05,
    iat_other=IAT_OTHER * 1.05
)
results_increase = multiple_runs(scenario=experiment_increase,
                                 rep=150,
                                 beds_start=9,
                                 beds_end=17,
                                 warmup=WARM_UP_PERIOD)

results_increase['Scenario'] = '5% Increase in Admissions'

results_increase = results_increase.round({
    "Overall Delay Probability": 2,
})

results_increase["1 in very n Patients"] = results_increase["1 in very n Patients"].round(0).astype(int)
results_increase

# =============================================================================
# Streamlit App Layout - Still a work in progress
# =============================================================================
st.set_page_config(page_title="Acute Stroke Unit Simulation", layout="wide")

# Main sidebar navigation
main_selection = st.sidebar.selectbox("Main Navigation", ["Home", "Simulation", "What If Scenario Analysis"])

if main_selection == "Home":
    st.title("Acute Stroke Unit Simulation")
    st.markdown("""    
        This app simulates patient flow through an Acute Stroke Unit using a Discrete Event Simulation model.
        The inspiration for this model comes from the [paper](https://doi.org/10.1186/s12913-016-1789-4).
        
        **Overview:**
        - Simulates multiple types of patient arrivals using exponential distributions.
        - Models Length Of Stay using lognormal distributions.
        - Tracks bed occupancy and delay probabilities.
        
        Use the sidebar to navigate between pages and adjust simulation settings.
    """)
    st.markdown(""" 
        The simplified model which this simulation includes is as seen below:
    """)
    st.image("images/replicated_model.png", caption="Simplified ASU Model", use_container_width=False, width=500)
    
elif main_selection == "Simulation":
    st.title("Simulation")

    # Sidebar controls
    st.sidebar.header("Simulation Settings")
    asu_beds_input = st.sidebar.slider("Number of ASU Beds", min_value=5, max_value=30, value=ASU_BEDS, step=1)
    run_length_input = st.sidebar.slider("Simulation Run Length (days)", min_value=100, max_value=3650, value=RUN_LENGTH, step=100)
    warmup_input = st.sidebar.slider("Warm-Up Period (days)", min_value=0, max_value=1000, value=WARM_UP_PERIOD, step=50)
    replication_mode = st.sidebar.radio("Run Mode", options=["Single Replication", "Multiple Replications"])
    n_reps_input = 1
    if replication_mode == "Multiple Replications":
        n_reps_input = st.sidebar.slider("Number of Replications", min_value=1, max_value=20, value=DEFAULT_N_REPS)

    # Run Simulation button
    if st.sidebar.button("Run Simulation"):
        st.subheader("Running Simulation...")

        #exp = Experiment(asu_beds=asu_beds_input)
        exp = Experiment(asu_beds=asu_beds_input, random_number_set=DEFAULT_RND_SET)
        experiment = Experiment()

        if replication_mode == "Single Replication":
            st.write(f"Seed used: {exp.random_number_set}")
            
            experiment_single = single_run(exp, run_length=run_length_input, warm_up=warmup_input)        

            # --- Patient Type Counts Summary ---
            patient_counts = {
                "Stroke": experiment_single.results["n_stroke"],
                "TIA": experiment_single.results["n_tia"],
                "Complex Neuro": experiment_single.results["n_complex_neuro"],
                "Other": experiment_single.results["n_other"],
                "Total Patients": experiment_single.results["n_patients"]
            }
            st.subheader("Patient Type Counts")
            st.table(pd.DataFrame.from_dict(patient_counts, orient="index", columns=["Count"]))


            # --- Transfer Breakdown by Destination ---
            transfer_data = {
                "Rehab": [
                    experiment_single.results["stroke_transfer"]["rehab"],
                    experiment_single.results["tia_transfer"]["rehab"],
                    experiment_single.results["complex_neuro_transfer"]["rehab"],
                    experiment_single.results["other_transfer"]["rehab"]
                ],
                "ESD": [
                    experiment_single.results["stroke_transfer"]["esd"],
                    experiment_single.results["tia_transfer"]["esd"],
                    experiment_single.results["complex_neuro_transfer"]["esd"],
                    experiment_single.results["other_transfer"]["esd"]
                ],
                "Other": [
                    experiment_single.results["stroke_transfer"]["other"],
                    experiment_single.results["tia_transfer"]["other"],
                    experiment_single.results["complex_neuro_transfer"]["other"],
                    experiment_single.results["other_transfer"]["other"]
                ]
            }

            transfer_df = pd.DataFrame(transfer_data, index=["Stroke", "TIA", "Complex Neuro", "Other"])
            transfer_df["Total Transfers"] = transfer_df.sum(axis=1)

            st.subheader("Transfer Destinations")
            st.table(transfer_df)

            # --- Delay Probabilities ---
            delay_p = exp.auditor.compute_delay_prob(warmup=warmup_input)
            st.subheader("Delay Probabilities")
            delay_df = pd.DataFrame.from_dict(delay_p["type"], orient="index", columns=["Probability"])
            delay_df.loc["overall"] = delay_p["overall"]
            st.table(delay_df)

            # --- Bed Utilisation ---
            bed_util = experiment_single.auditor.compute_bed_utilization(warmup=warmup_input)
            st.subheader("Bed Utilisation")
            bed_util_df = pd.DataFrame([bed_util])
            st.table(bed_util_df)

        else:
            results_df = multiple_replications(
                exp,
                rc_period=run_length_input,
                n_reps=n_reps_input,
                warmup=warmup_input
            )
            st.subheader("Summary of Replications")
            st.dataframe(results_df)           

            # --- Optional warm-up plot (uses one extra replication) ---
            exp_for_plot = Experiment(asu_beds=asu_beds_input)
            _ = single_run(exp_for_plot, run_length=run_length_input, warm_up=warmup_input)

            bed_data_df = pd.DataFrame(exp_for_plot.auditor.delay_time, columns=["type", "delay", "Timestamp"])
            
            fig = plot_warmup_analysis(bed_data_df, warm_up_period=warmup_input, detailed=False)
            st.subheader("Warm-Up Analysis Plot")
            st.pyplot(fig)

elif main_selection == "What If Scenario Analysis":
    st.title("What If Scenario Analysis")
    st.markdown("""
        The following predefined scenarios simulate variations in patient admission rates and patient types.
        These parameters are fixed and cannot be modified by users for consistency in comparison.
    """)
    scenario_choice = st.radio("Select Scenario", options=["Current Admissions", "5% Increase in Admissions", "No Complex-Neurological Cases"])
    if st.button("Run Scenario Analysis"):
        warmup_input = WARM_UP_PERIOD
        n_reps_input = 150

        if scenario_choice == "Current Admissions":
            exp_default = Experiment(asu_beds=ASU_BEDS)
            results_default = multiple_runs(exp_default, rep=n_reps_input, beds_start=9, beds_end=17, warmup=warmup_input)
            results_default['Scenario'] = 'Current Admissions'
            st.write("#### Scenario: Current Admissions")
            st.dataframe(results_default)
            combined_results = results_default

        elif scenario_choice == "5% Increase in Admissions":
            exp_increase = Experiment(
                iat_stroke=IAT_STROKE * 1.05,
                iat_tia=IAT_TIA * 1.05,
                iat_complex_neuro=IAT_COMPLEX_NEURO * 1.05,
                iat_other=IAT_OTHER * 1.05,
                asu_beds=ASU_BEDS
            )
            results_increase = multiple_runs(exp_increase, rep=n_reps_input, beds_start=9, beds_end=17, warmup=warmup_input)
            results_increase['Scenario'] = '5% Increase in Admissions'
            st.write("#### Scenario: 5% Increase in Admissions")
            st.dataframe(results_increase)
            combined_results = results_increase

        elif scenario_choice == "No Complex-Neurological Cases":
            exp_exclusion = Experiment(
                iat_complex_neuro=float('inf'),
                asu_beds=ASU_BEDS
            )
            results_exclusion = multiple_runs(exp_exclusion, rep=n_reps_input, beds_start=9, beds_end=17, warmup=warmup_input)
            results_exclusion['Scenario'] = 'No Complex-Neurological Cases'
            st.write("#### Scenario: No Complex-Neurological Cases")
            st.dataframe(results_exclusion)
            combined_results = results_exclusion

        st.write("#### Consolidated Scenario Results")
        st.dataframe(combined_results)
        combined_results.to_csv("result/scenarios.csv")