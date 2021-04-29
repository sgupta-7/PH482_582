#%%
import gizmo_analysis as gizmo
import utilities as ut
import os
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import ImageNormalize, LogStretch
import umap
from time import time
from sklearn.manifold import TSNE
from module_plot_cluster import ang_mom, ang_mom_circ, density_plot

#%%
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 13   # could be smaller
matplotlib.rcParams['axes.titlesize'] = 20   # could be smaller
#%%
path = '/Users/meh/Desktop/programs/ph582_programs/ph582_final_project/MLclass-master/m12f'
# path = '/Volumes/Seagate Expansion Drive/fire/m12f'
part = gizmo.io.Read.read_snapshots(['star'], 'index', 600,
                                    simulation_directory=path,
                                    assign_host_principal_axes=True,
                                    assign_formation_coordinates=True)
# path_out = '~/PycharmProjects/MLclass/'
#%%
def plot_results(data, c, clabel, xlabel='t-SNE 2d - one', ylabel='t-SNE 2d - two', **kw):
    if 'clim' in kw:
        vmin, vmax = kw['clim']
    else:
        vmin, vmax = [np.min(c), np.max(c)]
    if data.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(data[:, 0], data[:, 1], c=c, cmap='Spectral', s=1, vmin=vmin, vmax=vmax)
        plt.gca().set_aspect('equal', 'datalim')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(label=clabel)
        if 'method' in kw:
            plt.title(kw['method'], fontsize=11)
        if 'fname' in kw:
            plt.tight_layout()
            plt.savefig(kw['fname']+'.png', dpi=120)
        plt.show()
        plt.close()
#%%  filter stars (sample A or sample B)
idxs_star = np.where((part['star'].prop('host.distance.principal.total')>30) & (part['star'].prop('host.distance.principal.total')<500))
print(len(idxs_star[0]))
#%%  define arrays
k = 100
feh = part['star'].prop('metallicity.fe')[idxs_star][::k]
ofe = part['star'].prop('metallicity.o-metallicity.fe')[idxs_star][::k]
mgfe = part['star'].prop('metallicity.mg-metallicity.fe')[idxs_star][::k]
sife = part['star'].prop('metallicity.si-metallicity.fe')[idxs_star][::k]
cafe = part['star'].prop('metallicity.ca-metallicity.fe')[idxs_star][::k]
cfe = part['star'].prop('metallicity.c-metallicity.fe')[idxs_star][::k]
nefe = part['star'].prop('metallicity.ne-metallicity.fe')[idxs_star][::k]
nfe = part['star'].prop('metallicity.n-metallicity.fe')[idxs_star][::k]
sfe = part['star'].prop('metallicity.s-metallicity.fe')[idxs_star][::k]
ages = part['star'].prop('age')[idxs_star][::k]
x = part['star'].prop('host.distance.principal').T[0][idxs_star][::k]
y = part['star'].prop('host.distance.principal').T[1][idxs_star][::k]
z = part['star'].prop('host.distance.principal').T[2][idxs_star][::k]
R = np.sqrt(x**2 + y**2 + z**2)
vx = part['star'].prop('host.velocity.principal').T[0][idxs_star][::k]
vy = part['star'].prop('host.velocity.principal').T[1][idxs_star][::k]
vz = part['star'].prop('host.velocity.principal').T[2][idxs_star][::k]
vR = part['star'].prop('host.velocity.principal.cylindrical').T[0][idxs_star][::k]
vphi = part['star'].prop('host.velocity.principal.cylindrical').T[1][idxs_star][::k]
Rbirth = part['star'].prop('form.host.distance.total')[idxs_star][::k]
xb = part['star'].prop('form.host.distance.principal').T[0][idxs_star][::k]
yb = part['star'].prop('form.host.distance.principal').T[1][idxs_star][::k]
zb = part['star'].prop('form.host.distance.principal').T[2][idxs_star][::k]
#%%  angular momentum
j = ang_mom([x, y, z], [vx, vy, vz])
phi = part['star'].prop('potential')[idxs_star][::k]
v = np.sqrt(vx**2 + vy**2 + vz**2)
E = v**2 + phi
# j_circ = ang_mom_circ(R, v, phi)
#%% ang mom ratio histogram
plt.figure()
plt.hist(j[2]/E, bins=30) # , range=(-1, 1))
plt.savefig('jz_E_no_feh.png')
plt.show()
plt.close()
#%% 2d histogram of distribution
density_plot(part['star'].prop('host.distance.principal').T[0][idxs_star],
             part['star'].prop('host.distance.principal').T[1][idxs_star],
             xlabel='x', ylabel='y', name='xy_200kpc.png', axlim=[[-200, 200], [-200, 200]],
             clim=[1, 30], bins=512)

#%% 2d hist + age
density_plot(part['star'].prop('host.distance.principal').T[0][idxs_star],
             part['star'].prop('host.distance.principal').T[1][idxs_star],
             c=part['star'].prop('age')[idxs_star],
             xlabel='x', ylabel='y', name='xy_age_200kpc.png', axlim=[[-200, 200], [-200, 200]],
             clim=[0, 14], stretch='lin', bins=512, cmap='Spectral', clabel='age')
#%% 2d hist + Rb
density_plot(part['star'].prop('host.distance.principal').T[0][idxs_star],
             part['star'].prop('host.distance.principal').T[2][idxs_star],
             c=part['star'].prop('form.host.distance.principal.total')[idxs_star],
             xlabel='x', ylabel='z', name='xz_Rb_200kpc.png', axlim=[[-200, 200], [-200, 200]],
             clim=[60, 100], stretch='lin', bins=512, cmap='Spectral', clabel='$\mathrm{R_{birth}}$')

#%%  scatter plot in coords space
plt.figure()
plt.scatter(part['star'].prop('host.distance.principal').T[0][idxs_star][::100],
             part['star'].prop('host.distance.principal').T[1][idxs_star][::100], s=1, c=Rbirth)
plt.colorbar()
plt.xlim(-550, 550)
plt.ylim(-550, 550)
plt.savefig('scatter_coord_550kpc.png', dpi=120)
plt.show()
plt.close()

#%% 2d hist FeH-OFe
density_plot(part['star'].prop('metallicity.fe')[idxs_star],
             part['star'].prop('metallicity.o-metallicity.fe')[idxs_star],
             xlabel='[Fe/H]', ylabel='[O/Fe]', name='feh_ofe_80kpc.png', axlim=[[-4, 1.2], [0.2, 0.8]],
             clim=[1, 150], bins=512)

#%% 2d hist FeH-OFe-Rbirth
density_plot(part['star'].prop('metallicity.fe')[idxs_star],
             part['star'].prop('metallicity.o-metallicity.fe')[idxs_star],
             c=part['star'].prop('form.host.distance.principal.total')[idxs_star],
             xlabel='[Fe/H]', ylabel='[O/Fe]', name='feh_ofe_Rb_80kpc.png', axlim=[[-4, 1.2], [0.2, 0.8]],
             clim=[50, 110], title=' ', clabel='$\mathrm{R_{birth}, ~kpc}$', bins=512, figsize=(16, 14), cmap='Spectral')
#%% age histogram
plt.figure(figsize=(12, 8))
plt.hist(part['star'].prop('age')[idxs_star], bins=100)
plt.xlabel('age, Gyr')
plt.ylabel('N')
plt.savefig('age_hist_550kpc.png', dpi=120)
plt.close()

#%% umap
data_subset = np.array([feh, ofe, mgfe, cafe, sife, nfe, nefe, sfe, cfe]).T
start = time()
reducer = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.01, metric='minkowski', n_components=2)
reducer.fit(data_subset)
print(time()-start)

embedding = reducer.transform(data_subset)

plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1], c=Rbirth, cmap='Spectral', s=1)
plt.gca().set_aspect('equal', 'datalim')
plt.savefig('umap_sample_b_500kpc_newhp.png', dpi=120)
plt.show()
plt.close()

#%%
plot_results(embedding, Rbirth, '$\mathrm{R_{birth}, kpc}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_500kpc_Rbirth', clim=[0, 200])
#%%
plot_results(embedding, ages, '$\mathrm{age, Gyr}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_500kpc_age', clim=[0, 14])

#%%
plot_results(embedding, feh, '$\mathrm{[Fe/H]}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_200kpc_feh_no_feh', clim=[-2., 1.])
#%%
plot_results(embedding, ofe, '$\mathrm{[O/Fe]}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_500kpc_ofe_no_feh', clim=[0.45, 0.5])

#%%
plot_results(embedding, mgfe, '$\mathrm{[Mg/Fe]}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_500kpc_mgfe_no_feh', clim=[0.2, 0.4])

#%%
plot_results(embedding, cafe, '$\mathrm{[Ca/Fe]}$',method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_500kpc_cafe_no_feh', clim=[-0.1, 0.])
#%%
plot_results(embedding, sife, '$\mathrm{[Si/Fe]}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_500kpc_sife_no_feh', clim=[0.2, 0.4])

#%%
plot_results(embedding, sfe, '$\mathrm{[S/Fe]}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_500kpc_sfe_no_feh', clim=[0.2, 0.3])

#%%
plot_results(embedding, nfe, '$\mathrm{[N/Fe]}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_500kpc_nfe_no_feh', clim=[-0.25, 0.3])

#%%
plot_results(embedding, nefe, '$\mathrm{[Ne/Fe]}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_200kpc_nefe_no_feh', clim=[0.4, 0.6])

#%%
plot_results(embedding, cfe, '$\mathrm{[C/Fe]}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_500kpc_cfe_no_feh', clim=[0.2, 0.4])
#%%
plot_results(embedding, np.abs(vz), '$\mathrm{|v_z|, ~km/s}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_200kpc_vz_no_feh', clim=[0, 100])

#%%
plot_results(embedding, R, '$\mathrm{R, ~kpc}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_200kpc_R_no_feh', clim=[0, 200])

#%%
plot_results(embedding, np.abs(j[2]/np.max(j[2])), '$\mathrm{|J_{z}|}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_200kpc_absjz_.1max_no_feh', clim=[0., .1])

#%%
plot_results(embedding, np.abs(E/np.max(E)), '$\mathrm{|E|}$', method=str(reducer),
             xlabel='umap 2d - one', ylabel='umap 2d - one',
             fname='umap_chem_100_200kpc_absE_.1max_no_feh', clim=[0., .1])

#%% 3d plot of results
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=Rbirth, cmap='Spectral', s=1.)
ax.view_init(30, 80)
# plt.gca().set_aspect('equal', 'datalim')
plt.show()
plt.close()
#%%
plt.figure()
plt.scatter(xb, zb, s=.6, c=Rbirth)
plt.xlim(-1000, 1000)
plt.ylim(-1000, 1000)
plt.show()
plt.close()

#%%
groupII_umap = np.asarray((ages < 6)&(embedding[:, 1] < embedding[:, 0]*(-3.)+30.), dtype=int)
groupI_umap = np.asarray((ages > 7)&(embedding[:, 1] < embedding[:, 0]*5.), dtype=int)
groupIIb_umap = np.asarray((embedding[:, 1] > embedding[:, 0]*(-3.)+30.)&(embedding[:,1]<embedding[:,0]-10), dtype=int)

#%%
plt.figure()
plt.scatter(x[np.where(groupI_umap == 1)], z[np.where(groupI_umap == 1)], s=0.7, alpha=0.7, color='coral', label='group I')
plt.scatter(x[np.where(groupII_umap == 1)], z[np.where(groupII_umap == 1)], s=0.3, alpha=0.7, color='mediumturquoise', label='group II')
plt.scatter(x[np.where(groupIIb_umap == 1)], z[np.where(groupIIb_umap == 1)], s=0.3, alpha=0.7, color='navy', label='group IIb')
plt.legend()
plt.xlim(-20, 20)
plt.xlabel('X, kpc')
plt.ylabel('Z, kpc')
plt.ylim(-20, 20)
# plt.savefig('XZ_group_20kpc_umap.png', dpi=120)
plt.show()
plt.close()

#%%
plt.figure()
# plt.scatter(embedding[:, 0], embedding[:, 1], s=1.)
plt.scatter(embedding[:, 0][np.where(groupI_umap == 1)], embedding[:, 1][np.where(groupI_umap == 1)], s=0.7, alpha=0.7, color='coral', label='group I')
plt.scatter(embedding[:, 0][np.where(groupII_umap == 1)], embedding[:, 1][np.where(groupII_umap == 1)], s=0.3, alpha=0.7, color='mediumturquoise', label='group II')
plt.scatter(embedding[:, 0][np.where(groupIIb_umap == 1)], embedding[:, 1][np.where(groupIIb_umap == 1)], s=0.3, alpha=0.7, color='navy', label='group IIb')
plt.legend()
plt.xlim(-8, 20)
# plt.xlabel('X, kpc')
# plt.ylabel('Z, kpc')
plt.ylim(-7, 17)
plt.savefig('umap_res_group_200kpc_umap.png', dpi=120)
plt.show()
plt.close()
#%% t-SNE
data_subset = np.array([feh, ofe, mgfe, cafe, sife, nfe, nefe, sfe, cfe]).T  #[np.where(np.abs(j[2]/j_circ) < 0.3)]
#%%
time_start = time()
tsne = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=500, learning_rate=1000,
            random_state=42)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time()-time_start))
#%%
# time_start = time()
# tsne_large = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=500, learning_rate=1000,
#             random_state=42)
# tsne_results_large = tsne_large.fit_transform(data_subset)
# print('t-SNE done! Time elapsed: {} seconds'.format(time()-time_start))
#%%
# len(idxs_star[0][::1000])
plt.figure()
plt.hist(Rbirth, bins=100)
plt.savefig('Rbirth_hist_500kpc.png')
plt.close()
#%%
plot_results(tsne_results, ages, 'age, Gyr', method=str(tsne), clim=[0., 14.], fname='tsne_chem_100_500kpc_age')
#%%
plot_results(tsne_results, Rbirth, '$\mathrm{R_{birth}, kpc}$', method=str(tsne),
             fname='tsne_chem_100_500kpc_Rbirth', clim=[0, 200])

#%%
plot_results(tsne_results, feh, '$\mathrm{[Fe/H]}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_feh_no_feh', clim=[-2., 0.5])
#%%
plot_results(tsne_results, ofe, '$\mathrm{[O/Fe]}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_ofe_no_feh', clim=[0.45, 0.5])

#%%
plot_results(tsne_results, mgfe, '$\mathrm{[Mg/Fe]}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_mgfe_no_feh', clim=[0.2, 0.4])

#%%
plot_results(tsne_results, cafe, '$\mathrm{[Ca/Fe]}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_cafe_no_feh', clim=[-0.1, 0.])
#%%
plot_results(tsne_results, sife, '$\mathrm{[Si/Fe]}$', method=str(tsne),
             fname='tsne_chem_10_80kpc_sife_no_feh', clim=[0.2, 0.4])

#%%
plot_results(tsne_results, sfe, '$\mathrm{[S/Fe]}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_sfe_no_feh', clim=[0.2, 0.3])

#%%
plot_results(tsne_results, nfe, '$\mathrm{[N/Fe]}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_nfe', clim=[-0.25, 0.3])

#%%
plot_results(tsne_results, nefe, '$\mathrm{[Ne/Fe]}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_nefe_no_feh', clim=[0.4, 0.6])

#%%
plot_results(tsne_results, cfe, '$\mathrm{[C/Fe]}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_cfe_no_feh', clim=[0.2, 0.4])
#%%
plot_results(tsne_results, np.abs(vz), '$\mathrm{|v_z|, ~km/s}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_vz_p300_lr1e4_no_feh', clim=[0, 100])

#%%
plot_results(tsne_results, R, '$\mathrm{R, ~kpc}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_R_no_feh', clim=[0, 200])

#%%
plot_results(tsne_results, np.abs(j[2]/np.max(j[2])), '$\mathrm{|J_{z}|}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_absjz_.1max_no_feh', clim=[0., .1])

#%%
plot_results(tsne_results, np.abs(E/np.max(E)), '$\mathrm{|E|}$', method=str(tsne),
             fname='tsne_chem_100_200kpc_absE_.1max_no_feh', clim=[0., .1])

#%%
groupII_tsne = np.asarray(tsne_results[:, 1] > 37., dtype=int)
groupI_tsne = np.asarray((tsne_results[:, 1] < 37)&(tsne_results[:, 0] < 20)&(ages > 6), dtype=int)
groupIII_tsne = np.asarray((tsne_results[:, 1] < 37)&(tsne_results[:, 0] > -40)&(ages < 6), dtype=int)

#%%
plt.figure()
plt.scatter(tsne_results[:, 0][groupII_tsne == 1], tsne_results[:, 1][groupII_tsne == 1], s=1., label='II', color='navy')
plt.scatter(tsne_results[:, 0][groupI_tsne == 1], tsne_results[:, 1][groupI_tsne == 1], s=1., label='I', color='coral')
plt.scatter(tsne_results[:, 0][groupIII_tsne == 1], tsne_results[:, 1][groupIII_tsne == 1], s=1., label='III', color='mediumturquoise')
# plt.axhline(37.)
plt.savefig('tsne_group_200kpc.png', dpi=120)
plt.legend()
plt.show()
plt.close()
#%%
plt.figure()
plt.scatter(x[np.where(groupI_tsne == 1)], z[np.where(groupI_tsne == 1)], s=0.7, alpha=0.7, color='coral', label='group I')
plt.scatter(x[np.where(groupIII_tsne == 1)], z[np.where(groupIII_tsne == 1)], s=0.3, alpha=0.7, color='mediumturquoise', label='group III')
plt.scatter(x[np.where(groupII_tsne == 1)], z[np.where(groupII_tsne == 1)], s=0.3, alpha=0.7, color='navy', label='group II')
plt.legend()
plt.xlim(-20, 20)
plt.xlabel('X, kpc')
plt.ylabel('Z, kpc')
plt.ylim(-20, 20)
plt.savefig('XZ_group_200kpc.png', dpi=120)
plt.show()
plt.close()
#%%
id_tsne_groupI = part['star'].prop('id')[idxs_star][::k][groupI_tsne == 1]
id_tsne_groupII = part['star'].prop('id')[idxs_star][::k][groupII_tsne == 1]
id_tsne_groupIII = part['star'].prop('id')[idxs_star][::k][groupIII_tsne == 1]
id_umap_groupI = part['star'].prop('id')[idxs_star][::k][groupI_umap == 1]
id_umap_groupII = part['star'].prop('id')[idxs_star][::k][groupII_umap == 1]
id_umap_groupIIb = part['star'].prop('id')[idxs_star][::k][groupIIb_umap == 1]
#%%
id_sph = np.asarray(list(set(id_tsne_groupI)&set(id_umap_groupI)))
id_thick = np.asarray(list(set(id_tsne_groupIII)&set(id_umap_groupII)))
id_thin = np.asarray(list(set(id_tsne_groupII)&set(id_umap_groupIIb)))

#%%
print(len(id_sph)/len(id_tsne_groupI), len(id_sph)/len(id_umap_groupI))
print(len(id_thick)/len(id_tsne_groupIII), len(id_thick)/len(id_umap_groupII))
print(len(id_thin)/len(id_tsne_groupII), len(id_thin)/len(id_umap_groupIIb))
#%% kmeans
from sklearn.cluster import KMeans
from sklearn.manifold import LocallyLinearEmbedding
#%%
kmeans = KMeans(n_clusters=6)
kmeans.fit(tsne_results)
kmeans_res = kmeans.predict(tsne_results)
plot_results(tsne_results, kmeans_res, 'kmeans result')
#%%
lle = LocallyLinearEmbedding()
lle_res = lle.fit_transform(tsne_results)
plot_results(tsne_results, lle_res, 'lle result')
#%%
plt.figure()
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=1.)
xs1 = np.linspace(-60, 20, 80)
ys1 = -np.sqrt(np.abs(40**2 - (xs1+20)**2))
plt.plot(xs1, ys1)
plt.show()
plt.close()

#%%
import itertools
fig = plt.figure(figsize=(20, 14))
gs = GridSpec(nrows=3, ncols=3, width_ratios=[1, 1, 1], wspace=0.25)
ax = [fig.add_subplot(gs[i, j]) for i, j in itertools.product([0, 1, 2], [0, 1, 2])]
i = 0
j = 0
k = 0
# print(ax[1])
for elem, elnm in zip([cafe, sife, mgfe, ofe, sfe, nfe, nefe, cfe], ['[Ca/Fe]', '[Si/Fe]', '[Mg/Fe]', '[O/Fe]', '[S/Fe]', '[N/Fe]', '[Ne/Fe]', '[C/Fe]']):
    ax[k].hist(elem, bins=128)
    # plt.gca().legend()
    ax[k].set_xlabel(elnm, fontsize=13)
    # i, j = list(itertools.product([0, 1, 2], [0, 1, 2]))[k]
    k += 1
plt.savefig(f'abund_hist.png')
plt.close()
#%%
plt.close('all')
#%%
feh
#%%
plt.figure()
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=R, cmap='Spectral', s=1, vmin=0, vmax=20)
plt.gca().set_aspect('equal', 'datalim')
plt.show()
plt.close()

#%%
plt.figure()
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=np.abs(vz), cmap='Spectral', s=1, vmin=0, vmax=120)
plt.gca().set_aspect('equal', 'datalim')
plt.show()
plt.close()
#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], c=ages, cmap='Spectral', s=1.)
ax.view_init(elev=30, azim=180)
# plt.gca().set_aspect('equal', 'datalim')
plt.show()
plt.close()

#%% DBSCAN
from sklearn.cluster import DBSCAN
#%%
start = time()
data_subset = np.array([feh, ofe, mgfe, cafe, sife, nfe, nefe, sfe, cfe]).T
clustering = DBSCAN(eps=0.01, min_samples=5).fit(data_subset)
print(time()-start)
print(len(set(clustering.labels_)))

#%%
plt.figure()
plt.scatter(x[clustering.labels_>1], z[clustering.labels_>1], c=clustering.labels_[clustering.labels_>1], s=1.)
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.show()
plt.close()
#%%
plt.figure()
plt.hist(clustering.labels_, bins=71)
plt.show()
plt.close()

#%%
a = np.vstack([x, y, z, j[2], E, tsne_results[:, 0], tsne_results[:, 1], embedding[:, 0], embedding[:, 1]])
np.savetxt("/Users/meh/Desktop/programs/ph582_programs/ph582_final_project/MLclass-master/sampleb_x_y_z_jz_E_tsne_umap.csv", a.T, delimiter=',')

