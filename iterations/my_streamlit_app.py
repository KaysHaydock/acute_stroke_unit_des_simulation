from distribution import Exponential, Lognormal, Bernoulli

import numpy as np
import itertools
import simpy

import streamlit as st

st.title("Acute Stroke Unit Simulation")

st.write("""
HPDM097 - Making a Difference with Health Data.

This simulation models patient flow through an Acute Stroke Unit (ASU). 
It captures patient arrivals, bed allocation, treatment durations and 
transfers
. 
Patients are categorised into stroke, transient ischemic attack (TIA), 
complex neurological cases and 'other' conditions, each with distinct 
inter-arrival times and length-of-stay distributions. 

The simulation tracks key metrics, including bed occupancy, patient wait times 
and transfer destinations, providing insights into resource utilization 
and patient pathways.
""")

st.image("images/streamlit.png",
         caption="Acute Stroke Unit Simulation Flow",
         use_container_width=True
)


# default mean inter-arrival times(exp)
IAT_STROKE = 1.2
IAT_TIA = 9.3
IAT_COMPLEX_NEURO = 3.6
IAT_OTHER = 3.2

##############################################################################
# MODIFICATION: create a sidebar for sliders
#with st.sidebar:
    # run variables (units = days)
#    ASU_BEDS = 10
with st.sidebar:    
    ASU_BEDS = st.sidebar.number_input("Number of ASU Beds", min_value=1, value=10)    
    
    RUN_LENGTH = st.sidebar.slider("Simulation Run Length (days)", min_value=1, max_value=100, value=20)
    
    TRACE = st.sidebar.checkbox("Enable Trace Logging", value=False)  
##############################################################################



# Default Length of Stay (LOS) parameters (mean, stdev for Lognormal distribution
LOS_STROKE = (7.4, 8.6)
LOS_TIA = (1.8, 2.3)
LOS_COMPLEX_NEURO = (4.0, 5.0)
LOS_OTHER = (3.8, 5.2)
LOS_STROKE_NESD = (7.4, 8.6)
LOS_STROKE_ESD = (4.6, 4.8)
LOS_STROKE_MORTALITY = (7.0, 8.7)

# % where patients go after the Acute Stroke Unit (ASU)
TRANSFER_PROBABILITIES = {
    "stroke": {"rehab": 0.24, "esd": 0.13, "other": 0.63},
    "tia": {"rehab": 0.01, "esd": 0.01, "other": 0.98},
    "complex_neuro": {"rehab": 0.11, "esd": 0.05, "other": 0.84},
    "other": {"rehab": 0.05, "esd": 0.10, "other": 0.85}}



# sampling settings, 4 for arrivals, 4 for LOS and for 4 transfer probabilities
N_STREAMS = 12
DEFAULT_RND_SET = 0

# Boolean switch to simulation results as the model runs
#TRACE = False


# run variables (units = days)
#RUN_LENGTH = 20



def trace(msg):
    """
    Turing printing of events on and off.

    Params:
    -------
    msg: str
        string to print to screen.
    """
    if TRACE:
        #print(msg)
        #st.text(msg)
        st.code(msg, language="plaintext")


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
        Initialize default parameters.
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

        # Initialize results storage
        self.init_results_variables()

        # Initialize sampling distributions (RNGs)
        self.init_sampling()

    def set_random_no_set(self, random_number_set):
        """
        Controls the random sampling by re-seeding.
        """
        self.random_number_set = random_number_set
        self.init_sampling()

    def init_sampling(self):
        """
        Creates the distributions used by the model and initializes
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
        Initializes all the experiment variables used in results collection.
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
    Represents a patient in the system.
    """
    def __init__(self, patient_id, env, args, acute_stroke_unit, patient_type):
        self.patient_id = patient_id
        self.env = env
        self.args = args
        self.acute_stroke_unit = acute_stroke_unit
        self.patient_type = patient_type
        self.waiting_time = 0.0  # track how long the patient waited for a bed

    def treatment(self):
        """
        Simulates the patient’s treatment process.
        Upon discharge, calls `transfer()` to determine next destination.
        """
        arrival_time = self.env.now
        los_distribution = self.args.los_distributions[self.patient_type]

        # Arrival message
        trace(f"Patient {self.patient_id} ({self.patient_type.upper()}) arrives at {arrival_time:.2f}.")

        # Request bed from the ASU
        with self.acute_stroke_unit.beds.request() as request:
            yield request
            self.waiting_time = self.env.now - arrival_time
            los = los_distribution.sample()

            # Bed assigned message
            trace(f"Patient {self.patient_id} ({self.patient_type.upper()}) gets a bed at {self.env.now:.2f}."
                  f" (Waited {self.waiting_time:.2f} days)")

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
    """

    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.beds = simpy.Resource(env, capacity=args.asu_beds)

    def patient_arrivals(self, patient_type, arrival_distribution):
        """
        A generator that creates patients of 'patient_type' according to
        the given 'arrival_distribution' (Exponential).
        """
        for patient_count in itertools.count(start=1):
            inter_arrival_time = arrival_distribution.sample()
            yield self.env.timeout(inter_arrival_time)

            # Track patient count
            self.args.results[f"n_{patient_type}"] += 1
            self.args.results["n_patients"] += 1

            trace(f"{self.env.now:.2f}: {patient_type.upper()} arrival.")

            # Create new patient and start its treatment process
            new_patient = Patient(patient_count, self.env, self.args, self, patient_type)
            self.env.process(new_patient.treatment())


def single_run(experiment,
               rep=0,
               run_length=RUN_LENGTH):
    """
    Perform a single run of the model and return the results.

    Parameters
    ----------
    experiment : Experiment
        The experiment/parameters to use with model
    rep : int
        The replication number (used to set random seeds).
    run_length : float
        The run length of the model in days (default = 3650 = 10 years).
    """
    # 1. Reset results for each run
    experiment.init_results_variables()

    # 2. Set the random number set for this run
    experiment.set_random_no_set(rep)

    # 3. Create a fresh environment and an AcuteStrokeUnit
    env = simpy.Environment()
    asu = AcuteStrokeUnit(env, experiment)

    # 4. Create patient arrival processes for different types of patients
    env.process(asu.patient_arrivals("stroke", experiment.arrival_stroke))
    env.process(asu.patient_arrivals("tia", experiment.arrival_tia))
    env.process(asu.patient_arrivals("complex_neuro", experiment.arrival_complex_neuro))
    env.process(asu.patient_arrivals("other", experiment.arrival_other))

    # 5. Run the simulation
    env.run(until=run_length)

    # 6. Trace summary of total patients
    total_patients = sum(experiment.results[key] for key in experiment.results if key.startswith("n_"))
    trace(f"Final summary for rep={rep}: {total_patients} total patients.")

    # Return the results dictionary
    return experiment.results


#TRACE = False
#experiment = Experiment()
#results = single_run(experiment)
#results

# Streamlit button to run the simulation
if st.button("Run Simulation"):
    with st.spinner("Simulation running..."):  
        experiment = Experiment(asu_beds=ASU_BEDS)
        results = single_run(experiment)

        # Display results in Streamlit
        st.success("Done!")
        st.subheader("Simulation Results")
        st.write("Total Patients: ", results["n_patients"])
        st.write("Stroke Patients: ", results["n_stroke"])
        st.write("TIA Patients: ", results["n_tia"])
        st.write("Complex Neuro Patients: ", results["n_complex_neuro"])
        st.write("Other Patients: ", results["n_other"])
        
        st.write("Transfers to Rehab: ", results["total_transfers"]["rehab"])
        st.write("Transfers to ESD: ", results["total_transfers"]["esd"])
        st.write("Transfers to Other: ", results["total_transfers"]["other"])