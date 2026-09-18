# %%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd


# %%
folders = ['V_1_K_.1_allostery', 'V_1_K_10_allostery',
           'V_.1_K_1_allostery', 'V_10_K_1_allostery']

st_variant = 'C2'


# %%
fig, axs = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(4, 3))

# ---- Plot data into each axis ----
for folder, ax in zip(folders, axs.flatten()):
    try:
        beta, mi_allo,S,P = np.loadtxt('sweep/'+folder+'_'+st_variant+'_res/A_MI.csv').T

        ax.plot(beta, mi_allo, label='Allosteric', color='C0', linewidth=2)


        beta, mi_nonallo,S,P = np.loadtxt('sweep/'+folder+'_'+st_variant+'_res/nonallo_A_MI.csv').T
        ax.plot(beta, mi_nonallo, label='Non-allosteric', color='C1', linewidth=2)

        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), useOffset=True)
        ax.yaxis.get_offset_text().set_position((0, 1.01))
        ax.yaxis.get_offset_text().set_fontsize(8)
        #ax.axvline(beta[mi_allo.argmax()], color='gray', linestyle='--', linewidth=1)
        #print(beta[mi_allo.argmax()])
        #print(mi_allo.max())
        _, V, _ , K, _ = folder.split('_')
        ax.set_xlim(0, 14)
    except:
        pass


# --- Axis labels ---
for ax in axs[-1]:
    ax.set_xlabel(r'$\beta/\gamma_S$', fontsize=12)
    ax.set_ylim(0)
for ax in axs.flatten():
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0)
    #reduce ticksize
    ax.tick_params(axis='both', which='major', labelsize=8)

fig.supylabel(r'Mutual Information ($\text{MI}_{AB}$)', fontsize=12, x=0.19, y=0.5)

fig.subplots_adjust(bottom=0.01)
#axs[0,0].set_xlim(beta[0], 14)
plt.tight_layout(rect=[0.07,0.05,0.98,0.98])

fig.legend(*axs[-1][0].get_legend_handles_labels(),loc='lower center', 
           bbox_to_anchor=(0.6, -0.01), ncol=2, frameon=False,
           handlelength=1,
           fontsize=10)

titles = [r' low $\xi_K$', r'high $\xi_K$',
          r' low $\xi_V$', r'high $\xi_V$']

for ax, title in zip(axs.flat, titles):
    ax.text(0.3, 1.1, title,transform=ax.transAxes, fontsize=10)
axs[0,0].set_yticks(np.array((1,3))*1e-3)
axs[1,0].set_yticks([1e-2])
plt.savefig('f2', dpi=1200, bbox_inches='tight')
plt.savefig('f2.svg', dpi=1200, bbox_inches='tight', transparent=True)
plt.show()


# %%
folders = ['V_?_K_1_allostery','V_1_K_?_allostery','V_?_K_1_allostery_nu10','V_1_K_?_allostery_nu10']


# %%

def plot_variable(folder, axs):
    betas, allos, MI, S, P = np.loadtxt(
        'fixed/' + folder + '/' + st_variant + '_fcases/B_report.csv'
    ).T
    
    for b in [5, 10, 15]:
        mask = betas == b

        axs[0].plot(
            allos[mask],
            MI[mask],
            label=rf'$\beta/\gamma_S$ = {int(b)}',
        )
        axs[1].plot(allos[mask], S[mask])
        axs[2].plot(allos[mask], P[mask])

    axs[2].set_xscale('log')
    axs[2].xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: rf'$10^{{{int(np.log10(x))}}}$'
        )
    )

    axs[0].set_ylim(0)


fig, axs = plt.subplots(3, 2, sharex=True, figsize=(4, 4.5))

for folder, axi in zip(folders[:2], axs.T):
    plot_variable(folder, axi)


# Titles and x-labels
axs[-1, 0].set_xlabel(r'$\xi_V$', fontsize=12)
axs[-1, 1].set_xlabel(r'$\xi_K$', fontsize=12)


# Y-labels on first column
for ax, label in zip(
    axs[:, 0],
    [r'MI$_{AB}$', r'$\langle S \rangle$', r'$\langle P \rangle$'],
):
    ax.set_ylabel(label, fontsize=12)


# Scientific notation above each y-axis
for ax in axs.flat:
    ax.set_xlim(1e-3, 1e3)

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))

    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(
        style='sci',
        axis='y',
        scilimits=(0, 0),
        useOffset=True,
    )

    ax.yaxis.get_offset_text().set_position((0, 1.01))
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis='both', which='major', labelsize=8)


axs[0,1].legend(*axs[0,0].get_legend_handles_labels(), fontsize=8, handlelength=1)

yticks = [[1e-2],[1e-2],[10],[10],[2,4],[1]]
for ax, yt in zip(axs.flatten(), yticks):
    ax.set_yticks(yt)
plt.tight_layout(rect=[0.07,0.05,0.98,0.98])
plt.savefig('f3.png', dpi=1200, bbox_inches='tight')
plt.savefig('f3.svg', dpi=1200, bbox_inches='tight', transparent=True)
plt.show()


# %%
records = []

files = os.listdir("varallostery/"+st_variant)
for file in files:
    try:
        allo, v, k, beta, shape, times, _ = file.split("_")
        xi_v, xi_k = v.split("=")[-1], k.split("=")[-1]

        # remove trailing 'K' from xi_v
        xi_v = float(xi_v)
        xi_k = float(xi_k)
        try:
            times = float(times)
        except:
            times = times

        records.append({
            "kind": allo,
            "xi_v": xi_v,
            "xi_k": xi_k,
            "shape": shape,
            "times": times,
            "beta": beta,
            "file": file
        })
    except ValueError:
        print(f"Skipping file (unexpected format): {file}")

# Convert to DataFrame
df = pd.DataFrame(records)



# %%
def make_plot(df, axs,tmax =-1):
    for i, row in df.iterrows():
        file = row['file']
        xi_v = row['xi_v']
        xi_k = row['xi_k']
        #times = row['times']
        
        t,betas,S,P,MI = np.loadtxt('varallostery/'+st_variant+'/'+file).T
        if tmax > 0:
            t,betas,S,P,MI = t[t<=tmax],betas[t<=tmax],S[t<=tmax],P[t<=tmax],MI[t<=tmax] 

        axs[0].plot(t, betas, linewidth = 2.5, color='k')
        axs[1].plot(t, MI)
        axs[2].plot(t,S)

        dic_label = {1:'1',.1:'0.1',10:'10'}
        if np.all(df['xi_k'] == xi_k):
            #axs[0].set_title(r'$\xi_K$ = {} variable $\xi_V$'.format(xi_k),fontsize=15)
            axs[3].plot(t, P, label=rf'$\xi_V$ = '+dic_label[xi_v])
        elif np.all(df['xi_v'] == xi_v):
            #axs[0].set_title(r'$\xi_V$ = {} variable $\xi_K$'.format(xi_v),fontsize=15)
            axs[3].plot(t, P, label=rf'$\xi_K$ = {dic_label[xi_k]}')

    axs[-1].set_xlim(t[0],t[-1])
    [ax.tick_params(axis='both', which='major', labelsize=8) for ax in axs.flatten()]


# %%

step_k1    = df[(df['kind']=='Allosteric')    & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_k']==1)  & df['xi_v'].isin([0.1, 1, 10])].sort_values('xi_v')
step_v1    = df[(df['kind']=='Allosteric')    & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_v']==1)  & df['xi_k'].isin([0.1, 1, 10])].sort_values('xi_k')

nastep_k1  = df[(df['kind']=='nonAllosteric') & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_k']==1)  & df['xi_v'].isin([0.1, 1, 10])].sort_values('xi_v')
nastep_v1  = df[(df['kind']=='nonAllosteric') & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_v']==1)  & df['xi_k'].isin([0.1, 1, 10])].sort_values('xi_k')


# %%
fig, axs = plt.subplots(4, 2, sharex=True,  figsize=(3.8,6))

for (df, axi) in zip([step_v1, step_k1], axs.T):
    make_plot(df, axi)

[ax.set_ylabel(g,fontsize=12) for (ax,g) in zip (axs.T[0],[r'$\beta/\gamma_S$',
                                                           r'MI$_{AB}$',
                                                           r'$\langle S \rangle$',
                                                           r'$\langle P \rangle$',])]
#axs[0,0].set_title(r'Silent K-allostery ($\xi_K$ = 1)',fontsize=15)
#axs[0,1].set_title(r'Silent V-allostery ($\xi_V$ = 1)'.format(xi_v),fontsize=15)
for ax in axs[1:].flatten():
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_position((0, 1.01))  # move exponent above axis

    ymin, ymax = ax.get_ylim()
    m = max(abs(ymin), abs(ymax))
    n = int(np.floor(np.log10(m)))

    if n>0:
        base = 3 * (10 ** n)      # 0, 3, 6 ... × 10^n
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=base))
        ax.set_ylim(0,np.ceil(m/10)*10)
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    #axs[0].set_title(r'$\xi_k$ = 10 variable $\xi_V$',fontsize=15)
[axi.set_xlabel(r'time',fontsize=12) for axi in axs[-1]]
lines , labels = [], []

fig.legend(*axs[-1][0].get_legend_handles_labels(),loc='lower center', 
           bbox_to_anchor=(0.55, 0.02), ncol=3, frameon=False,fontsize=10,
           handlelength=1)


axs[1,0].set_yticks([0,2e-3,4e-3, 6e-3])
#axs[0,0].set_xticks([0,40,80])

[axi.set_title(st,fontsize=10) for axi, st in zip(axs[0], ['K allosteric', 'V allosteric'])]
plt.tight_layout(rect=[0.07,0.05,0.98,0.98])
fig.savefig('f4.png',bbox_inches='tight',dpi=1200)
fig.savefig('f4.svg',bbox_inches='tight',transparent=True)
 

# %%
fig, axs = plt.subplots(4, 2, sharex=True, sharey='row', figsize=(3.8,6))

for (df, axi) in zip([step_v1, nastep_v1], axs.T):
    make_plot(df, axi)

[ax.set_ylabel(g,fontsize=12) for (ax,g) in zip (axs.T[0],[r'$\beta/\gamma_S$',
                                                           r'MI$_{AB}$',
                                                           r'$\langle S \rangle$',
                                                           r'$\langle P \rangle$',])]
#axs[0,0].set_title(r'Silent K-allostery ($\xi_K$ = 1)',fontsize=15)
#axs[0,1].set_title(r'Silent V-allostery ($\xi_V$ = 1)'.format(xi_v),fontsize=15)
for ax in axs[1:].flatten():
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_position((0, 1.01))  # move exponent above axis

    ymin, ymax = ax.get_ylim()
    m = max(abs(ymin), abs(ymax))
    n = int(np.floor(np.log10(m)))

    if n>0:
        base = 3 * (10 ** n)      # 0, 3, 6 ... × 10^n
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=base))
        ax.set_ylim(0,np.ceil(m/10)*10)
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    #axs[0].set_title(r'$\xi_k$ = 10 variable $\xi_V$',fontsize=15)
[axi.set_xlabel(r'time',fontsize=12) for axi in axs[-1]]
lines , labels = [], []

fig.legend(*axs[-1][0].get_legend_handles_labels(),loc='lower center', 
           bbox_to_anchor=(0.55, 0.02), ncol=3, frameon=False,fontsize=10,
           handlelength=1)


axs[1,0].set_yticks([0,2e-3,4e-3, 6e-3])
axs[0,0].set_xticks([0,40,80])

[axi.set_title(st,fontsize=10) for axi, st in zip(axs[0], ['K allosteric', 'Non-allosteric'])]
plt.tight_layout(rect=[0.07,0.05,0.98,0.98])
fig.savefig('f4_V_comparison.png',bbox_inches='tight',dpi=1200)
fig.savefig('f4_V_comparison.svg',bbox_inches='tight',transparent=True)


# %%
fig, axs = plt.subplots(4, 2, sharex=True, sharey='row', figsize=(3.8,6))

for (df, axi) in zip([step_k1, nastep_k1], axs.T):
    make_plot(df, axi)

[ax.set_ylabel(g,fontsize=12) for (ax,g) in zip (axs.T[0],[r'$\beta/\gamma_S$',
                                                           r'MI$_{AB}$',
                                                           r'$\langle S \rangle$',
                                                           r'$\langle P \rangle$',])]
#axs[0,0].set_title(r'Silent K-allostery ($\xi_K$ = 1)',fontsize=15)
#axs[0,1].set_title(r'Silent V-allostery ($\xi_V$ = 1)'.format(xi_v),fontsize=15)
for ax in axs[1:].flatten():
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_position((0, 1.01))  # move exponent above axis

    ymin, ymax = ax.get_ylim()
    m = max(abs(ymin), abs(ymax))
    n = int(np.floor(np.log10(m)))

    if n>0:
        base = 3 * (10 ** n)      # 0, 3, 6 ... × 10^n
        ax.yaxis.set_major_locator(ticker.MultipleLocator(base=base))
        ax.set_ylim(0,np.ceil(m/10)*10)
    ax.yaxis.get_offset_text().set_fontsize(8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    #axs[0].set_title(r'$\xi_k$ = 10 variable $\xi_V$',fontsize=15)
[axi.set_xlabel(r'time',fontsize=12) for axi in axs[-1]]
lines , labels = [], []

fig.legend(*axs[-1][0].get_legend_handles_labels(),loc='lower center', 
           bbox_to_anchor=(0.55, 0.02), ncol=3, frameon=False,fontsize=10,
           handlelength=1)

axs[1,0].set_yticks([0,1e-2,2e-2])
axs[2,0].set_yticks([0,5])
axs[2,0].set_ylim(0,10)
axs[0,0].set_xticks([0,40,80])

[axi.set_title(st,fontsize=10) for axi, st in zip(axs[0], ['V allosteric', 'Non-allosteric'])]
plt.tight_layout(rect=[0.07,0.05,0.98,0.98])
fig.savefig('f4_K_comparison.png',bbox_inches='tight',dpi=1200)
fig.savefig('f4_K_comparison.svg',bbox_inches='tight',transparent=True)


# %%



