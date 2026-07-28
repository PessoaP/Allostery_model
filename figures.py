# %%
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd


# %%
folders = ['V_1_K_.1_allostery', 'V_1_K_1_allostery', 'V_1_K_10_allostery',
           'V_.1_K_1_allostery', 'V_1_K_1_allostery', 'V_10_K_1_allostery']


st_variant = 'C2'
variants_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
os.makedirs('figs', exist_ok=True)
[os.makedirs('figs/'+variant, exist_ok=True) for variant in variants_list]

for variant in variants_list:
    folders_sweep = ['sweep/'+f+'_'+variant+'_res' for f in folders]

    fig, axs = plt.subplots(2, 3, sharex=True, sharey='row', figsize=(8, 5))
    # ---- Plot data into each axis ----
    for folder, ax in zip(folders_sweep, axs.flatten()):
        try:
            beta, mi_allo,S,P = np.loadtxt(folder+'/A_MI.csv').T

            ax.plot(beta, mi_allo, label='Allosteric', color='C0', linewidth=2)


            beta, mi_nonallo,S,P = np.loadtxt(folder+'/nonallo_A_MI.csv').T
            ax.plot(beta, mi_nonallo, label='Non-allosteric', color='C1', linewidth=2)

            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_position((0, 1.01))
            ax.axvline(beta[mi_allo.argmax()], color='gray', linestyle='--', linewidth=1)
            #print(beta[mi_allo.argmax()])
            #separate V and K rate in folder name

            parts = os.path.basename(folder).split('_')
            V, K = parts[1], parts[3]

            ax.set_title(rf'$\xi_V = {V},\ \xi_K = {K}$', fontsize=11)
            ax.set_xlim(0, 14)
            # ax.text(
            #     0.98, 0.95,
            #     rf'$\xi_V={V},\ \xi_K={K}$',
            #     transform=ax.transAxes,
            #     fontsize=12,
            #     va='top',
            #     ha='right',
            #     bbox=dict(facecolor='white', alpha=0.75, edgecolor='0.8', linewidth=0.5, pad=2)
            # )
        except:
            pass

    axs.flatten()[-1].legend()

    # --- Axis labels ---
    for ax in axs[-1]:
        ax.set_xlabel(r'$\beta/\gamma_S$')
        ax.set_ylim(0)
    for ax in axs.flatten():
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0)

    fig.supylabel(r'Mutual Information ($\text{MI}_{AB}$)' + (( ' -- ' + variant) if st_variant != variant else ''),
                  x=0.1, y=0.5)

    fig.subplots_adjust(bottom=0.01)
    #axs[0,0].set_xlim(beta[0], 14)
    plt.tight_layout(rect=[0.07,0.05,0.98,0.98])

    plt.savefig('figs/'+variant+'/f2.png', dpi=600, bbox_inches='tight')
    plt.savefig('figs/'+variant+'/f2.svg',  bbox_inches='tight')
    #plt.show()


# %%
variants_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
for variant in variants_list:
    folders_sweep = ['sweep/'+f+'_'+variant+'_res' for f in folders]

    fig, axs = plt.subplots(2, 3, sharex=True, sharey='row', figsize=(9, 5))

    # ---- Plot data into each axis ----
    for folder, ax in zip(folders_sweep, axs.flatten()):
        try:
            beta, mi_allo,S,P = np.loadtxt(folder+'/A_MI.csv').T

            ax.plot(beta, P, label='Allosteric', color='C0', linewidth=2)


            beta, mi_nonallo,S,P = np.loadtxt(folder+'/nonallo_A_MI.csv').T
            ax.plot(beta, P, label='Non-allosteric', color='C1', linewidth=2)

            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_position((0, 1.01))
            ax.axvline(beta[mi_allo.argmax()], color='gray', linestyle='--', linewidth=1)
            #print(beta[mi_allo.argmax()])
            #separate V and K rate in folder name
            
            parts = os.path.basename(folder).split('_')
            V, K = parts[1], parts[3]

            ax.set_title(rf'$\xi_V$ = {V}$ , \xi_K$ = {K}', fontsize=12)
            ax.set_xlim(0, 14)

        except:
            pass

    axs.flatten()[2].legend()



    # --- Axis labels ---
    for ax in axs[-1]:
        ax.set_xlabel(r'$\beta/\gamma_S$', fontsize=12)
        ax.set_ylim(0)
    for ax in axs.flatten():
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0)

    fig.supylabel(r'$\langle P \rangle$' + (( ' -- ' + variant) if st_variant != variant else ''), fontsize=16, x=0.1, y=0.5)

    fig.subplots_adjust(bottom=0.01)
    axs[0,0].set_xlim(beta[0], 10)
    plt.tight_layout(rect=[0.07,0.05,0.98,0.98])

    plt.savefig('figs/'+variant+'/f2p.png', dpi=600, bbox_inches='tight')
    plt.savefig('figs/'+variant+'/f2p.svg',  bbox_inches='tight')
    #plt.show()


# %%
variants_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
for variant in variants_list:
    folders_sweep = ['sweep/'+f+'_'+variant+'_res' for f in folders]

    fig, axs = plt.subplots(2, 3, sharex=True, sharey='row', figsize=(9, 5))

    # ---- Plot data into each axis ----
    for folder, ax in zip(folders_sweep, axs.flatten()):
        try:
            beta, mi_allo,S,P = np.loadtxt(folder+'/A_MI.csv').T

            ax.plot(beta, S, label='Allosteric', color='C0', linewidth=2)


            beta, mi_nonallo,S,P = np.loadtxt(folder+'/nonallo_A_MI.csv').T
            ax.plot(beta, S, label='Non-allosteric', color='C1', linewidth=2)

            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_position((0, 1.01))
            ax.axvline(beta[mi_allo.argmax()], color='gray', linestyle='--', linewidth=1)
            #print(beta[mi_allo.argmax()])
            #separate V and K rate in folder name
            parts = os.path.basename(folder).split('_')
            V, K = parts[1], parts[3]
            ax.set_title(rf'$\xi_V$ = {V}$ , \xi_K$ = {K}', fontsize=12)
            ax.set_xlim(0, 14)

        except:
            pass

    axs.flatten()[2].legend()



    # --- Axis labels ---
    for ax in axs[-1]:
        ax.set_xlabel(r'$\beta/\gamma_S$', fontsize=12)
        ax.set_ylim(0)
    for ax in axs.flatten():
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0)

    fig.supylabel(r'$\langle S \rangle$' + (( ' -- ' + variant) if st_variant != variant else ''), fontsize=16, x=0.1, y=0.5)

    fig.subplots_adjust(bottom=0.01)
    axs[0,0].set_xlim(beta[0], 10)
    plt.tight_layout(rect=[0.07,0.05,0.98,0.98])

    plt.savefig('figs/'+variant+'/f2s.png', dpi=600, bbox_inches='tight')
    plt.savefig('figs/'+variant+'/f2s.svg',  bbox_inches='tight')
    #plt.show()


# %%
def plot_variable(folder,axs):
    betas, allos, MI, S, P = np.loadtxt(folder+'/B_report.csv').T
        #print(betas)
    for b in [5,10,15]:
        axs[0].plot(allos[betas == b], MI[betas == b], label=rf'$\beta/\gamma_S$ = {int(b)}')
        axs[1].plot(allos[betas == b], S[betas == b])
        axs[2].plot(allos[betas == b], P[betas == b])

    # Log scale for last subplot's x-axis
    axs[2].set_xscale('log')
    axs[2].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$')
        )
    axs[0].set_ylim(0) 

# %%
folders_fixed = ['V_?_K_1_allostery','V_1_K_?_allostery','V_?_K_1_allostery_nu10','V_1_K_?_allostery_nu10']


# %%
variants_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
for variant in variants_list:
    folders_fixed_variant = ['fixed/'+f+'/'+variant+'_fcases' for f in folders_fixed]


    for folder in folders_fixed_variant:
        fig, axs = plt.subplots(3, 1, sharex=True,figsize=(4,6))
        plot_variable(folder,axs)
        

        # Legend outside on top
        axs[0].legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1.2),
            ncol=3,
            frameon=False,fontsize=8
        )

        # Scientific notation with offset text (outside axis)
        for ax in axs:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.yaxis.get_offset_text().set_position((0, 1.01))  # move exponent above axis
    
        parts = folder.split('_')
        vrate, krate = parts[1], parts[3]

        if vrate == '?':
            axs[-1].set_xlabel(r'$\xi_V$',fontsize=15)
            axs[0].set_title(((variant + ' -- ') if st_variant != variant else '')+r'$\xi_K$ = {} variable $\xi_V$'.format(krate),fontsize=15)
        else:
            axs[-1].set_xlabel(r'$\xi_K$',fontsize=15)
            axs[0].set_title(((variant + ' -- ') if st_variant != variant else '')+r'$\xi_V$ = {} variable $\xi_K$'.format(vrate),fontsize=15)

        [ax.set_ylabel(g,fontsize=14) for (ax,g) in zip (axs,[r'MI$_{AB}$',
                                                r'$\langle S \rangle$',
                                                r'$\langle P \rangle$',])]

        
        plt.tight_layout()
        ##plt.show()
        plt.savefig('figs/'+variant+'/f3.png', dpi=600, bbox_inches='tight')
        plt.savefig('figs/'+variant+'/f3.svg',  bbox_inches='tight')


# %%
variants_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
for variant in variants_list:
    folders_fixed_variant = ['fixed/'+f+'/'+variant+'_fcases' for f in folders_fixed]

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 7))

    for (folder, axi) in zip(folders_fixed_variant[:2], axs.T):
        plot_variable(folder, axi)
            
    # Titles and x-labels
    #axs[0, 0].set_title(r'Silent K-allostery ($\xi_K$ = 1)', fontsize=20, pad=18)
    axs[-1, 0].set_xlabel(r'$\xi_V$', fontsize=20)
    #axs[0, 1].set_title(r'Silent V-allostery ($\xi_V$ = 1)', fontsize=20, pad=18)
    axs[-1, 1].set_xlabel(r'$\xi_K$', fontsize=20)

    # Y-labels on first column
    for ax, g in zip( axs[:, 0], [r'MI$_{AB}$' + ((' -- ' + variant) if st_variant != variant else ''), r'$\langle S \rangle$', r'$\langle P \rangle$']):
        ax.set_ylabel(g, fontsize=20)

    # === Manual "×10^exp" style on all panels (including 10^0) ===
    for ax in axs.flatten():
        ax.set_xlim(1e-3,1e3)
        #ax.set_xlim(1e-2,1e2)
        ymin, ymax = ax.get_ylim()
        scale = max(abs(ymin), abs(ymax))
        if scale == 0:
            exp = 0
        else:
            exp = int(np.floor(np.log10(scale)))

        ticks = ax.get_yticks()
        if len(ticks) > 0:
            max_tick = max(abs(t) for t in ticks)
            # If everything is < 1 but exp==0, shift exponent down so we don't get 0.2, 0.4, ...
            if exp == 0 and max_tick < 1 and max_tick > 0:
                exp -= 1

        factor = 10**exp

        # Rescale tick labels
        ticks = ax.get_yticks()  # get again in case they changed
        ax.set_yticklabels([f"{t / factor:g}" for t in ticks])

        # Put ×10^exp above the axis
        ax.text(
            0.0, 1.02,
            rf"$\times 10^{{{exp}}}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
        )
    # axs[0,1].text(0.83, 1.03, rf"$\nu = 10 \gamma_S$", transform=axs[0,1].transAxes)
    # axs[0,0].text(0.85, 1.03, rf"$\nu = 1 \gamma_S$", transform=axs[0,0].transAxes)

    # for ax in axs[0][1:]:
    #     ymin, ymax = ax.get_ylim()
    #     ax.set_ylim(ymin,ymax/6)

    # Legend (currently just on top-right axis; can switch to fig.legend if you want global)
    axs[-1, 0].legend(*axs[0,0].get_legend_handles_labels(),fontsize=14)
    #fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False,fontsize=15)

    fig.subplots_adjust(bottom=0.12)
    plt.tight_layout()
    plt.savefig('figs/'+variant+'/f3.png', dpi=600, bbox_inches='tight')
    plt.savefig('figs/'+variant+'/f3.svg',  bbox_inches='tight')
    #plt.show()


# %%
def get_records_from_files(folder = "varallostery"):
    records = []
    files = os.listdir(folder)
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
    return pd.DataFrame(records)



# %%
def make_plot(df, axs,tmax =-1,variant='C2'):
    for i, row in df.iterrows():
        file = row['file']
        xi_v = row['xi_v']
        xi_k = row['xi_k']
        #times = row['times']

        t,betas,S,P,MI = np.loadtxt('varallostery/'+ variant + '/' + file).T
        if tmax > 0:
            t,betas,S,P,MI = t[t<=tmax],betas[t<=tmax],S[t<=tmax],P[t<=tmax],MI[t<=tmax] 

        axs[0].plot(t, betas, linewidth = 5, color='k')
        axs[1].plot(t, MI)
        axs[2].plot(t,S)
        if np.all(df['xi_k'] == xi_k):
            #axs[0].set_title(r'$\xi_K$ = {} variable $\xi_V$'.format(xi_k),fontsize=15)
            axs[3].plot(t, P, label=rf'$\xi_V$ = {xi_v:.2f}')
        elif np.all(df['xi_v'] == xi_v):
            #axs[0].set_title(r'$\xi_V$ = {} variable $\xi_K$'.format(xi_v),fontsize=15)
            axs[3].plot(t, P, label=rf'$\xi_K$ = {xi_k:.2f}')

    axs[-1].set_xlim(t[0],t[-1])

# %%
for variant in variants_list:
    df = get_records_from_files(folder = "varallostery/"+variant)

    step_k1    = df[(df['kind']=='Allosteric')    & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_k']==1)  & df['xi_v'].isin([0.1, 1, 10])].sort_values('xi_v')
    step_v1    = df[(df['kind']=='Allosteric')    & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_v']==1)  & df['xi_k'].isin([0.1, 1, 10])].sort_values('xi_k')

    nastep_k1  = df[(df['kind']=='nonAllosteric') & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_k']==1)  & df['xi_v'].isin([0.1, 1, 10])].sort_values('xi_v')
    nastep_v1  = df[(df['kind']=='nonAllosteric') & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_v']==1)  & df['xi_k'].isin([0.1, 1, 10])].sort_values('xi_k')


    fig, axs = plt.subplots(4, 2, sharex=True,figsize=(8,7.5))

    for (df, axi) in zip([step_k1, step_v1], axs.T):
        make_plot(df, axi, variant=variant)

    [ax.set_ylabel(g,fontsize=14) for (ax,g) in zip (axs.T[0],[r'$\beta/\gamma_S$',
                                                            r'MI$_{AB}$',
                                                            r'$\langle S \rangle$',
                                                            r'$\langle P \rangle$',])]
    axs[0,0].set_title(((variant+' -- ') if st_variant != variant else '')+r'Silent K-allostery ($\xi_K$ = 1)',fontsize=15)
    axs[0,1].set_title(((variant+' -- ') if st_variant != variant else '')+r'Silent V-allostery ($\xi_V$ = 1)',fontsize=15)
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
        #axs[0].set_title(r'$\xi_k$ = 10 variable $\xi_V$',fontsize=15)
    [axi.set_xlabel(r'time',fontsize=15) for axi in axs[-1]]
    lines , labels = [], []

    fig.legend(*axs[-1][0].get_legend_handles_labels(),loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,fontsize=12)

    #fig.suptitle(variant, fontsize=16)
    fig.savefig('figs/'+variant+'/f4.png',bbox_inches='tight',dpi=600)
    fig.savefig('figs/'+variant+'/f4.svg',bbox_inches='tight')
    #plt.show()

# %%
for variant in variants_list:
    df = get_records_from_files(folder = "varallostery/"+variant)

    step_k1    = df[(df['kind']=='Allosteric')    & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_k']==1)  & df['xi_v'].isin([0.1, 1, 10])].sort_values('xi_v')
    step_v1    = df[(df['kind']=='Allosteric')    & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_v']==1)  & df['xi_k'].isin([0.1, 1, 10])].sort_values('xi_k')

    nastep_k1  = df[(df['kind']=='nonAllosteric') & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_k']==1)  & df['xi_v'].isin([0.1, 1, 10])].sort_values('xi_v')
    nastep_v1  = df[(df['kind']=='nonAllosteric') & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_v']==1)  & df['xi_k'].isin([0.1, 1, 10])].sort_values('xi_k')


    fig, axs = plt.subplots(4, 2, sharex=True, sharey='row', figsize=(8,7.5))

    for (df, axi) in zip([step_k1, nastep_k1], axs.T):
        make_plot(df, axi, variant=variant)

    [ax.set_ylabel(g,fontsize=14) for (ax,g) in zip (axs.T[0],[r'$\beta/\gamma_S$',
                                                            r'MI$_{AB}$',
                                                            r'$\langle S \rangle$',
                                                            r'$\langle P \rangle$',])]
    #axs[0,0].set_title(r'Silent K-allostery ($\xi_K$ = 1)',fontsize=15)
    #axs[0,1].set_title(r'Non-allosteric equiv'.format(xi_v),fontsize=15)
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
        #axs[0].set_title(r'$\xi_k$ = 10 variable $\xi_V$',fontsize=15)
    [axi.set_xlabel(r'time',fontsize=15) for axi in axs[-1]]

    fig.legend(*axs[-1][0].get_legend_handles_labels(),loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,fontsize=12)

    
    if st_variant != variant:
        fig.suptitle(((variant+' -- ') if st_variant != variant else '') + r'$\xi_K$ = 1 variable $\xi_V$', fontsize=16)
    fig.savefig('figs/'+variant+'/f4_V_comparison.png',bbox_inches='tight',dpi=250)
    fig.savefig('figs/'+variant+'/f4_V_comparison.svg',bbox_inches='tight')
    #plt.show()

# %%
for variant in variants_list:
    df = get_records_from_files(folder = "varallostery/"+variant)

    step_k1    = df[(df['kind']=='Allosteric')    & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_k']==1)  & df['xi_v'].isin([0.1, 1, 10])].sort_values('xi_v')
    step_v1    = df[(df['kind']=='Allosteric')    & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_v']==1)  & df['xi_k'].isin([0.1, 1, 10])].sort_values('xi_k')

    nastep_k1  = df[(df['kind']=='nonAllosteric') & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_k']==1)  & df['xi_v'].isin([0.1, 1, 10])].sort_values('xi_v')
    nastep_v1  = df[(df['kind']=='nonAllosteric') & (df['beta']=='10') & (df['shape']=='varstep') & (df['times']=='[14.  6.]') & (df['xi_v']==1)  & df['xi_k'].isin([0.1, 1, 10])].sort_values('xi_k')


    fig, axs = plt.subplots(4, 2, sharex=True, sharey='row', figsize=(8,7.5))

    for (df, axi) in zip([step_v1, nastep_v1], axs.T):
        make_plot(df, axi, variant=variant) 

    [ax.set_ylabel(g,fontsize=14) for (ax,g) in zip (axs.T[0],[r'$\beta/\gamma_S$',
                                                            r'MI$_{AB}$',
                                                            r'$\langle S \rangle$',
                                                            r'$\langle P \rangle$',])]
    #axs[0,0].set_title(r'Silent V-allostery ($\xi_K$ = 1)',fontsize=15)
    #axs[0,1].set_title(r'Non-allosteric equiv'.format(xi_v),fontsize=15)
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
        #axs[0].set_title(r'$\xi_k$ = 10 variable $\xi_V$',fontsize=15)
    [axi.set_xlabel(r'time',fontsize=15) for axi in axs[-1]]

    fig.legend(*axs[-1][0].get_legend_handles_labels(),loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False,fontsize=12)

    if st_variant != variant:
        fig.suptitle(((variant+' -- ') if st_variant != variant else '') + r'$\xi_V$ = 1 variable $\xi_K$', fontsize=16)
    fig.savefig('figs/'+variant+'/f4_K_comparison.png',bbox_inches='tight',dpi=400)
    fig.savefig('figs/'+variant+'/f4_K_comparison.svg',bbox_inches='tight')
    #plt.show()

# %%



