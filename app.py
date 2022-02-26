from flask import Flask, render_template, request

import aiq.state

from aiq.AIQ import load_samples, web_simple_mc_estimator
from aiq.agents.WebManual import WebManual
from aiq.agents.Freq import Freq
from aiq.refmachines.BF import BF

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html', current_iteration=aiq.state.CURRENT_ITERATION)


#background process happening without any refreshing
@app.route('/run_test', methods=['POST'])
def background_process_test():
    sample_size = request.form.get('sample_size')
    print("sample_size:       " + str(sample_size))
    agent = request.form.get('agent')
    print("agent:       " + str(agent))
    ref_mach = request.form.get('ref_mach')
    print("ref_mach:       " + str(ref_mach))

    _start(sample_size, agent, ref_mach)
    return render_template('results.html', current_iteration=aiq.state.CURRENT_ITERATION)


def _start(sample_size, agent, ref_mach):
    if agent == 'Manual':
        _runManual()
    else:
        _runTest()


def _runManual():
    cluster_node = ''
    proportion_of_total = 0.95
    agent_str = 'WebManual'
    episode_length = 2
    sample_size = 2
    simple_mc = True
    disc_rate = 1.0
    proportion_of_total = 1.0 - disc_rate ** episode_length

    refm_str = 'BF'
    refm_call = refm_str + "." + refm_str + "( )"
    refm = BF()

    agent_call = agent_str + "." + agent_str + "( refm, " + str(disc_rate) + ")"
    agent = WebManual(refm, 1.0)

    print("Reference machine:       " + str(refm))
    print("RL Agent:                " + str(agent))
    print("Discount rate:           " + str(disc_rate))
    print("Episode length:          " + str(episode_length))
    print("Sample size:             " + str(sample_size))

    samples, dist = load_samples(refm, cluster_node, simple_mc)

    web_simple_mc_estimator( refm_call, agent_call, episode_length, disc_rate, sample_size )

def _runTest():
    cluster_node = ''
    proportion_of_total = 0.95
    agent_str = 'Freq'
    episode_length = 2
    sample_size = 200
    simple_mc = True
    disc_rate = 1.0
    proportion_of_total = 1.0 - disc_rate ** episode_length

    refm_str = 'BF'
    refm_call = refm_str + "." + refm_str + "( )"
    refm = BF()

    agent_call = agent_str + "." + agent_str + "( refm, " + str(disc_rate) + ", " + str(disc_rate) + ")"
    agent = Freq(refm, 1.0, disc_rate)

    print("Reference machine:       " + str(refm))
    print("RL Agent:                " + str(agent))
    print("Discount rate:           " + str(disc_rate))
    print("Episode length:          " + str(episode_length))
    print("Sample size:             " + str(sample_size))

    samples, dist = load_samples(refm, cluster_node, simple_mc)

    print (web_simple_mc_estimator( refm_call, agent_call, episode_length, disc_rate, sample_size ))
