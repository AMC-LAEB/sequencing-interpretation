
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from matplotlib.ticker import LogLocator
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings
from google.colab.output import eval_js

wget_call = """
wget https://raw.githubusercontent.com/AMC-LAEB/sequencing-interpretation/main/inc_model.keras
wget https://raw.githubusercontent.com/AMC-LAEB/sequencing-interpretation/main/lr_model.keras
wget https://raw.githubusercontent.com/AMC-LAEB/sequencing-interpretation/main/inc_scaler.pkl
wget https://raw.githubusercontent.com/AMC-LAEB/sequencing-interpretation/main/lr_scaler.pkl
"""

def load_models():
  lr_model = load_model("lr_model.keras")
  lr_scaler = joblib.load("lr_scaler.pkl")
  inc_model = load_model("inc_model.keras")
  inc_scaler = joblib.load("inc_scaler.pkl")

def generateCurve(mutrate, k, r0, err_rate, genint, import_rate):
  df = pd.DataFrame({
    'mutrate': mutrate,
    'k': np.log(k),
    'r0': r0,
    'err_rate' : err_rate,
    'time_to_peak': np.arange(-200,200).tolist() * 3,
    'genint': genint,
    'import_rate': np.log(import_rate),
    'nmut': [0] * 400 + [1] * 400 + [2] * 400,
    })
  
  pred_inc = inc_model.predict(inc_scaler.transform(df[['r0','genint','time_to_peak']]),verbose=0,batch_size=64)
  df['inc_pred'] = pred_inc
  
  pred_lr = lr_model.predict(lr_scaler.transform(df[['mutrate', 'k', 'r0', 'err_rate', 'time_to_peak', 'genint', 'import_rate','nmut']]),verbose=0,batch_size=64)
  df['lr_pred'] = pred_lr

  df = df[df['inc_pred']>np.log(100)]
  colors = ["#FBBE22FF", "#CC4248FF", "#56106EFF"]

  fig, ax1 = plt.subplots(figsize=[2,1.5],dpi =300)

  window = 5
  labels = ["Identical","1 SNP","2 SNPs"]
  
  for nmut in [0,1,2]:
    lr_smooth = pd.Series(df[df['nmut']==nmut]['lr_pred']).rolling(window=window, center=True).mean()
    ax1.plot(df[df['nmut']==nmut]['time_to_peak'],np.exp(lr_smooth), color=colors[nmut],linewidth=1,label=labels[nmut])
  
  ax1.set_xlabel("Days post-peak",fontsize=5)
  ax1.set_ylabel("Likelihood ratio",fontsize=5)
  ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=None))  # major ticks at 10^n
  ax1.set_yscale('log', base=10)
  ax1.set_ylim([0.01,1e5])
  ax1.tick_params(axis='both', labelsize=4,width=0.1)
  ax1.axhline(y=1, color='darkgrey', linestyle='--', linewidth=.3)
  ax1.legend(fontsize=3)
  for spine in ax1.spines.values():
    spine.set_linewidth(0.1)  
  ax1.grid(which='major', linestyle='--', linewidth=0.1)

  ax2 = ax1.twinx()
  ax2.tick_params(axis='both', labelsize=4,width=0.1)
  inc_smooth = pd.Series(df['inc_pred']).rolling(window=window, center=True).mean()
  ax2.plot(df['time_to_peak'],np.exp(inc_smooth),linewidth=0.5)
  ax2.set_ylabel('Incidence',fontsize=5)
  ax2.set_ylim([0,4e4])
  for spine in ax2.spines.values():
    spine.set_linewidth(0.1)  

  fig.show()

def create_ui():
  warnings.filterwarnings('ignore')

  eval_js('google.colab.output.setIframeHeight("700")')
  style = {'description_width': '200px'}
  layout = widgets.Layout(min_width='700px')

  display(HTML('''
  <style>
  .widget-slider .ui-slider-horizontal {
    height: 12px !important;  /* increase thickness */
    margin-top: 15px !important; /* adjust vertical position if needed */
  }

  .widget-slider .ui-slider-horizontal .ui-slider-handle {
    top: -5px !important;  /* center handle vertically */
    height: 24px !important;  /* bigger handle */
    width: 24px !important;
    margin-left: -12px !important;
  }

  '''))

  r0 = widgets.FloatSlider(min=1.2, max=2.5, step=0.1, value=1.6, description='R0', style=style, layout=layout)
  genint = widgets.FloatSlider(min=2, max=10, step=1, value=3, description='Mean generation interval', style=style, layout=layout)
  k = widgets.FloatSlider(min=0.1, max=1, step=.1, value=1, description='Transmission overdispersion', style=style, layout=layout)
  err_rate = widgets.FloatSlider(min=0, max=.5, step=.1, value=0, description='Sequencing error rate', style=style, layout=layout)
  import_rate = widgets.FloatSlider(min=1, max=100, step=10, value=1, description='Number of initial introductions', style=style, layout=layout)
  mutrate = widgets.FloatSlider(min=0.5, max=3, step=.1, value=2, description='Mutation rate', style=style, layout=layout)

  out = widgets.Output()

  def update_plot(mutrate, k, r0, err_rate, genint, import_rate):
    with out:
        out.clear_output(wait=True)
        generateCurve(mutrate,k,r0,err_rate,genint,import_rate)

  ui = widgets.VBox([mutrate,k, r0, err_rate, genint, import_rate])

  ui.layout = widgets.Layout(width='700px',min_width='700px')
  out.layout = widgets.Layout(width='500px',min_width='500px')
  layout = widgets.VBox([ui, out],width='500px')

  interactive = widgets.interactive_output(
      update_plot,{
        "mutrate": mutrate,
      "k": k,
      "r0": r0,
      "err_rate": err_rate,
      "genint": genint,
      "import_rate": import_rate})
  
  return layout, interactive




