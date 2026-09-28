"""Build a measured-results report. Fails if the experiment evidence is missing."""
import json
import os
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4

RES = ROOT / 'results'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
AR = ['fvsbn','made_1layer','made_2layer','made_lr_low','made_small_data']
NAMES = {'fvsbn':'FVSBN','made_1layer':'MADE 1 layer','made_2layer':'MADE 2 layers',
         'made_lr_low':'MADE low LR','made_small_data':'MADE 10k data','gan':'GAN (0.9)',
    'gan_no_smoothing':'GAN (1.0)', 'made_long':'MADE 120 epochs','gan_long':'GAN 100 epochs'}


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def make_figures(metrics):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':160,'savefig.bbox':'tight'})
    fig, ax = plt.subplots(1,2,figsize=(10,3.5))
    for name in AR:
        h=read(RES/name/'history.json')
        ax[0].plot([r['epoch'] for r in h],[r['val_nll'] for r in h],label=NAMES[name])
    ax[0].set(xlabel='Epoch',ylabel='Validation NLL (nats/image)',title='Equal epoch budgets')
    ax[0].legend(fontsize=7)
    for name in ['fvsbn','made_2layer']:
        h=read(RES/name/'history.json')
        ax[1].plot([r['epoch'] for r in h],[r['train_nll'] for r in h],label=NAMES[name]+' train')
        ax[1].plot([r['epoch'] for r in h],[r['val_nll'] for r in h],'--',label=NAMES[name]+' validation')
    ax[1].set(xlabel='Epoch',ylabel='NLL (nats/image)',title='Optimization and generalization')
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG/'ar_curves.png'); plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    for name in ['gan','gan_no_smoothing']:
        h=read(RES/name/'history.json')
        ax[0].plot([r['epoch'] for r in h],[r['generator_loss'] for r in h],label=NAMES[name]+' G')
        ax[0].plot([r['epoch'] for r in h],[r['discriminator_loss'] for r in h],'--',label=NAMES[name]+' D')
        diag=read(RES/name/'diagnostics.json')
        shift=-.2 if name=='gan' else .2
        ax[1].bar(np.arange(10)+shift,np.array(diag['class_counts'])/20,.4,label=NAMES[name])
    ax[0].set(xlabel='Epoch',ylabel='Binary cross-entropy',title='Adversarial training'); ax[0].legend(fontsize=7)
    ax[1].set(xlabel='Predicted digit',ylabel='Generated samples (%)',title='Classifier-based coverage (n=2,000)',xticks=range(10))
    ax[1].axhline(10,color='gray',linestyle=':',linewidth=1); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG/'gan_curves.png'); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,4.7))
    for ax,name in zip(axes,['fvsbn','made_2layer','gan']):
        im=PILImage.open(RES/name/'progress.png')
        ax.imshow(im,interpolation='nearest')
        ax.set_title(NAMES[name]); ax.set_xticks([])
        h=im.height/5
        ax.set_yticks([(i+.5)*h for i in range(5)],['Epoch '+str(e) for e in metrics[name]['progress_epochs']],fontsize=8)
        for sp in ax.spines.values():sp.set_visible(False)
    fig.tight_layout();fig.savefig(FIG/'progress.png');plt.close(fig)


def main():
    metrics={name:read(RES/name/'metrics.json') for name in AR+['gan','gan_no_smoothing','made_long','gan_long']}
    diagnostics={name:read(RES/name/'diagnostics.json') for name in ['gan','gan_no_smoothing','gan_long']}
    env=read(RES/'environment.json'); meta=read(ROOT/'submission_metadata.json')
    make_figures(metrics)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#202a35')))
    styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=5))
    styles.add(ParagraphStyle(name='CaptionCustom',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=9,textColor=colors.HexColor('#495667')))
    styles.add(ParagraphStyle(name='EquationCustom',fontName='Helvetica',fontSize=10,leading=16,spaceBefore=4,spaceAfter=8,leftIndent=12))
    styles['Title'].fontSize=24;styles['Title'].leading=29;styles['Title'].textColor=colors.HexColor('#183b57')
    styles['Heading1'].fontSize=17;styles['Heading1'].leading=21;styles['Heading1'].textColor=colors.HexColor('#183b57')
    styles['Heading2'].fontSize=12;styles['Heading2'].textColor=colors.HexColor('#183b57')
    story=[]
    def p(text,style='BodyCustom'):story.append(Paragraph(text,styles[style]))
    def heading(s):p(s,'Heading1')
    def sub(s):p(s,'Heading2')
    def page():story.append(PageBreak())
    def table(rows,widths=None):
        converted=[[Paragraph(str(v),styles['SmallCustom']) for v in row] for row in rows]
        t=Table(converted,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eff5')),
          ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7f98ae')),
          ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d4dce3')),('LEFTPADDING',(0,0),(-1,-1),6),
          ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,9)])
    def picture(path,width):
        im=PILImage.open(path);return Image(str(path),width=width,height=width*im.height/im.width)
    def fig(path,width,caption):story.append(picture(path,width));p(caption,'CaptionCustom')
    def equation(text,width):
        path=FIG/('equation_'+str(len(story))+'.png')
        fig_eq=plt.figure(figsize=(10,0.8))
        fig_eq.text(.01,.35,text,fontsize=16)
        fig_eq.savefig(path,dpi=220,bbox_inches='tight',pad_inches=.05)
        plt.close(fig_eq)
        story.extend([picture(path,width),Spacer(1,10)])
    f,m,g=metrics['fvsbn'],metrics['made_2layer'],metrics['gan']
    dg=diagnostics['gan'];dn=diagnostics['gan_no_smoothing']
    best=min(AR,key=lambda n:metrics[n]['validation_nll'])

    p('Learning to Generate<br/>Handwritten Digits','Title')
    p('FVSBN, neural autoregression and adversarial learning','Heading2')
    p(f"CS5494 Topics in Generative AI | Assignment 1<br/><b>{escape(meta['name'])} | Student ID: {escape(meta['student_id'])}</b><br/>Individual submission")
    sub('1. Introduction')
    p('I implement three unconditional generative models on MNIST. First, a fully visible sigmoid belief network (FVSBN) models every binary pixel with logistic regression on preceding pixels. Second, a masked neural autoregressive model (MADE) adds nonlinear hidden layers while preserving the same probability factorization. Third, a convolutional GAN generates original grayscale images from Gaussian noise. My aim is to compare model capacity, sample quality, training behavior and computational cost, rather than judge the models by a single loss.')
    p(f"In these runs, FVSBN achieves {f['test_nll']:.2f} nats/image and the two-layer MADE achieves {m['test_nll']:.2f} on threshold-binarized test images. The GAN is evaluated using uncurated samples, training trajectories and classifier-based diversity diagnostics; its adversarial loss is not a likelihood and cannot be compared numerically with autoregressive NLL.")
    sub('2. Data and experimental protocol')
    table([['Item','Choice and reason'],['Dataset','Original MNIST: 60,000 official training images and 10,000 official test images. All images remain 28 x 28.'],['Split','A fixed permutation (seed 2026) partitions official training data into 55,000 training and 5,000 validation images. The official test set remains held out.'],['AR preprocessing','Scale bytes to [0,1], then set x = 1 when intensity >= 0.5 and 0 otherwise. Flatten in row-major raster order. This is fixed threshold binarization, not stochastic binarization.'],['GAN preprocessing','Use original grayscale values scaled to [-1,1]. No thresholding, resizing or augmentation. GAN trains on the same 55,000-image subset.'],['Labels','Not supplied to FVSBN, MADE, generator or discriminator. Labels are used only to train an independent diagnostic digit classifier.']], [100,399])
    p('All generative runs use seed 42. AR configurations use equal epoch budgets, batch size 256 and the same validation/test split. I select each AR checkpoint by minimum validation NLL and report its test NLL afterward. GAN evaluation uses the predefined final epoch. These choices separate model selection from held-out evaluation.','SmallCustom')

    page();heading('3. Autoregressive methods')
    sub('3.1 Joint distribution and objective')
    p('Following the chain-rule formulation in Lectures 2 and 3 [1,2], I order the D = 784 binary pixels from top left to bottom right. Each conditional is Bernoulli with probability q<sub>i</sub>:')
    equation(r'$p(\mathbf{x})=\prod_{i=1}^{D}p(x_i\mid x_{<i}),\quad p(x_i\mid x_{<i})=q_i^{x_i}(1-q_i)^{1-x_i}$',335)
    equation(r'$\mathrm{NLL}(\mathbf{x})=-\sum_{i=1}^{D}\left[x_i\log q_i+(1-x_i)\log(1-q_i)\right]$',305)
    p('I minimize the average image NLL with Adam, using binary_cross_entropy_with_logits for numerical stability. The loss is summed over pixels, then averaged over images. Bits per pixel = NLL/(784 ln 2). Training evaluates all conditionals in one masked forward pass using observed prefixes; generation must instead sample pixels sequentially from previously generated prefixes.')
    sub('3.2 FVSBN: linear conditional logits')
    equation(r'$q_i=\mathrm{sigmoid}\left(b_i+\sum_{j<i}W_{ij}x_j\right)$',175)
    p('A strictly lower-triangular mask on a 784 x 784 weight matrix enforces j &lt; i. There are 784 x 783 / 2 + 784 = 307,720 active parameters. The implementation stores the dense matrix for efficient batch operations, so it allocates 615,440 parameters; masked entries are inactive. The first pixel has only a bias. This is a generative joint model built from logistic conditionals, rather than a digit classifier. Its limitation is that each conditional logit is linear in the prefix.')
    sub('3.3 MADE: nonlinear conditional logits')
    p('I use MADE [3] as the neural autoregressive model. It follows the nonlinear conditional modeling idea discussed for NADE in Lecture 3, but uses explicit masks for parallel likelihood evaluation. The main network is 784 - 512 - 512 - 784 with ReLU hidden activations and Bernoulli output logits. A one-hidden-layer 784 - 512 - 784 network tests the effect of depth.')
    p('Input and output pixel i have degree i. A hidden unit k of width H has degree floor(k(D-1)/H)+1, for k = 0,...,H-1. A hidden connection is retained when the source degree is no greater than the destination degree. The output connection is retained only when the hidden degree is strictly less than the output degree. Every path into output i therefore starts at an input j &lt; i. Masks are fixed registered buffers and are saved in checkpoints.')
    p('All AR output biases start at the clamped empirical training pixel log-odds (probabilities clipped to [0.001,0.999]); other weights use PyTorch defaults. This accelerates fitting the mostly black background. During generation, I draw u<sub>i</sub> uniformly and set x<sub>i</sub> = 1[u<sub>i</sub> &lt; q<sub>i</sub>]. I reuse the same uniforms for progress grids, so changes reflect the model rather than a new sampling seed.')
    sub('3.4 Correctness checks')
    p('Tests verify zero gradients to present/future pixels, exact normalization on all four-pixel states, and agreement between 20,000 samples and enumerated probabilities. FVSBN sampling evaluates only the required weight row and prefix; MADE evaluates the full network at each pixel.','BodyCustom')

    page();heading('4. GAN method and evaluation')
    sub('4.1 Architecture')
    table([['Module','Layers and output shapes'],['Generator G','z ~ N(0,I), 64 dimensions; Linear 64 -> 6,272; BatchNorm1d + ReLU; reshape to 128 x 7 x 7; transposed convolution 128 -> 64 (kernel 4, stride 2, padding 1) -> 64 x 14 x 14; BatchNorm2d + ReLU; transposed convolution 64 -> 1 with the same settings -> 1 x 28 x 28; Tanh.'],['Discriminator D','Convolution 1 -> 32 (kernel 4, stride 2, padding 1) -> 32 x 14 x 14; LeakyReLU(0.2); convolution 32 -> 64 with the same settings -> 64 x 7 x 7; LeakyReLU(0.2); flatten; Linear 3,136 -> 1 logit. No output sigmoid in the loss implementation.']], [100,399])
    p(f"The generator has {g['generator_parameters']['stored']:,} parameters and the discriminator has {g['discriminator_parameters']['stored']:,}. The generator creates all pixels in one forward pass. Batch normalization operates in training mode for updates and evaluation mode for sampling. The GAN is unconditional and uses neither an encoder nor digit labels.")
    sub('4.2 Alternating optimization')
    p('I use the adversarial framework [4] with the non-saturating generator objective. Writing d(x) = sigmoid(D(x)), the implemented losses are:')
    p('L<sub>D</sub> = BCE(D(x), a) + BCE(D(G(z).detach()), 0)<br/>L<sub>G</sub> = BCE(D(G(z)), 1)','EquationCustom')
    p('For the main run, a = 0.9 applies one-sided smoothing to real discriminator labels; fake targets remain 0 and generator targets remain 1. A controlled ablation sets a = 1.0. Each minibatch performs one D update and one G update, with Adam learning rate 0.0002 and betas (0.5,0.999) for both networks. Fake images are detached for the D step; D parameters are frozen for the G step while gradients still flow through D to G. I reuse the generated batch between these two steps. Both runs use the same architecture, seed, data and 30-epoch budget.')
    sub('4.3 Stability, coverage and novelty diagnostics')
    p('I inspect generator/discriminator loss trajectories together with samples from fixed latent vectors. A classifier trained only on labeled training images supplies a diagnostic predicted-digit histogram for 2,000 fresh generated images. The classifier consists of two 3 x 3 convolution/ReLU/max-pool blocks (32 and 64 channels), then 3,136 -> 128 -> 10 linear layers with ReLU. It trains for five epochs using Adam (0.001), selecting the checkpoint by validation accuracy.')
    p('For class proportions p<sub>k</sub>, I report entropy H = -sum<sub>k</sub> p<sub>k</sub> ln p<sub>k</sub> and exp(H), the effective number of represented classes (maximum 10). I also report classifier confidence and nearest-training-image pixel RMSE for the first 100 generated samples against all 55,000 training images. These are diagnostic proxies: a confident classifier can misread artifacts, balanced class counts can hide within-class collapse, and pixel distance is not a perceptual similarity metric.')
    p('There is no claim of a directly comparable GAN likelihood or an ImageNet-based FID score. The generated samples and the conditional likelihood measurements answer different questions.','SmallCustom')

    page();heading('5. Quantitative results and ablations')
    rows=[['Model / change','Train n','LR','Best ep.','Val NLL','Test NLL (SE)','Bits/pixel']]
    for name in AR:
        r=metrics[name];rows.append([NAMES[name],f"{r['train_count']:,}",f"{r['learning_rate']:g}",r['best_epoch'],f"{r['validation_nll']:.2f}",f"{r['test_nll']:.2f} ({r['test_nll_se']:.2f})",f"{r['bits_per_pixel']:.4f}"])
    table(rows,[113,49,43,44,57,111,82])
    p('Table 1. Lower is better. NLL is in nats per 28 x 28 threshold-binarized image. Parentheses are standard errors across 10,000 test images, not uncertainty across training seeds. Every configuration has a 30-epoch budget; the best validation checkpoint may occur earlier.','CaptionCustom')
    a=np.load(RES/'fvsbn'/'test_nll.npy');b=np.load(RES/'made_2layer'/'test_nll.npy');delta=a-b
    p(f"The two-layer MADE changes test NLL by {m['test_nll']-f['test_nll']:+.2f} nats/image relative to FVSBN. The paired FVSBN-minus-MADE difference is {delta.mean():.2f} nats/image, with an approximate 95% interval [{delta.mean()-1.96*delta.std(ddof=1)/len(delta)**.5:.2f}, {delta.mean()+1.96*delta.std(ddof=1)/len(delta)**.5:.2f}] over test images for these fixed fitted models. This interval does not establish robustness to random initialization.")
    sub('Depth, learning rate and data quantity')
    one=metrics['made_1layer'];low=metrics['made_lr_low'];small=metrics['made_small_data']
    p(f"<b>Depth:</b> adding a second 512-unit layer changes validation NLL from {one['validation_nll']:.2f} to {m['validation_nll']:.2f} (difference {m['validation_nll']-one['validation_nll']:+.2f}). This is a practical depth comparison, not a parameter-matched ablation: the second layer also adds capacity and computation.")
    p(f"<b>Learning rate:</b> reducing Adam's step size from 0.001 to 0.0003 changes validation NLL from {m['validation_nll']:.2f} to {low['validation_nll']:.2f}. The interpretation is limited to this fixed epoch budget; a slower learning rate may require more training. The best validation configuration among the five is {NAMES[best]}.")
    p(f"<b>Data:</b> reducing the two-layer model's training set from 55,000 to 10,000 images changes validation NLL from {m['validation_nll']:.2f} to {small['validation_nll']:.2f}. The subset is nested and the held-out splits are identical. Equal epochs imply fewer optimizer steps for the smaller dataset, so this measures a practical data-and-compute reduction, not a pure data effect with matched update counts.")
    sub('Measured compute')
    rows=[['Model','Active / stored parameters','Train wall (s)','100 images (s)']]
    for name in ['fvsbn','made_1layer','made_2layer','gan']:
        r=metrics[name]
        pars=(f"{r['parameters']['active']:,} / {r['parameters']['stored']:,}" if name!='gan' else f"G {r['generator_parameters']['stored']:,}; D {r['discriminator_parameters']['stored']:,}")
        rows.append([NAMES[name],pars,f"{r['training_seconds']:.1f}",f"{r['inference']['seconds_median']:.4f}"])
    table(rows,[104,205,94,96])
    p(f"Table 2. Hardware: {escape(env['gpu'] or 'CPU')}; Python {env['python']}, PyTorch {env['torch']}. Wall time excludes setup/data loading but includes epoch validation and progress output. Inference excludes disk I/O and is the median of three GPU-synchronized runs after warm-up, batch size 100. Active AR parameters exclude masked entries. Update-only times and individual timing repeats are in metrics.json.",'CaptionCustom')

    page();heading('6. Learning curves and generated samples')
    fig(FIG/'ar_curves.png',499,'Figure 1. Validation curves for all AR configurations and training/validation curves for the two main AR models. Training NLL is averaged while weights change within an epoch; validation NLL is measured afterward, so their small differences are not a strict generalization-gap estimate.')
    t=Table([[picture(RES/'fvsbn'/'samples.png',231),picture(RES/'made_2layer'/'samples.png',231)]],colWidths=[249.5,249.5])
    story.append(t)
    p('Figure 2. First 100 uncurated samples from FVSBN (left) and two-layer MADE (right). Both use seed 1234 sampling uniforms and their validation-selected checkpoints. White pixels are 1 and black pixels are 0. These are sampled binary pixels, not probability heat maps.','CaptionCustom')
    p('The grids expose both successful and ambiguous digits. The likelihood comparison uses the same binary representation and exact normalized distributions, whereas visual inspection emphasizes recognizable stroke structure. I retain every sample in the fixed grid to avoid overstating quality through selection. Lower likelihood loss alone does not guarantee that every ancestral sample is recognizable.')

    page();heading('7. Generation during training')
    fig(FIG/'progress.png',499,'Figure 3. Five stages and four fixed samples per stage for each model. AR models reuse the same uniform random values; GAN reuses the same four latent vectors. Epoch 0 is the untrained initialization (AR biases already match training pixel marginals). Later rows show the actual saved epoch states, which can differ from a validation-selected final AR checkpoint.')
    fig(RES/'gan'/'samples.png',260,'Figure 4. First 100 uncurated images from the final main GAN checkpoint. Intensities are mapped from [-1,1] back to [0,1] for display; they are not binarized.')

    page();heading('8. GAN stability and diversity')
    fig(FIG/'gan_curves.png',499,'Figure 5. Loss trajectories and predicted-digit frequencies for two equally budgeted GAN runs. The dotted 10% line is a uniform reference, not an enforced class prior. Discriminator losses have different real targets between runs and are therefore not directly comparable measures of quality.')
    rows=[['Diagnostic','Real label 0.9','Real label 1.0']]
    for label,key,fmt in [('Class entropy (nats)','class_entropy_nats','.3f'),('Effective classes / 10','effective_classes','.2f'),('Mean maximum class probability','mean_max_class_probability','.3f'),('Fraction with confidence >= 0.9','fraction_confidence_ge_0_9','.3f'),('Mean nearest-train pixel RMSE','nearest_train_rmse_mean','.4f'),('Minimum nearest-train pixel RMSE','nearest_train_rmse_min','.4f')]:
        rows.append([label,format(dg[key],fmt),format(dn[key],fmt)])
    table(rows,[281,109,109])
    p(f"Table 3. Coverage/confidence use 2,000 fresh generated images per run (seed 999). Nearest-neighbor diagnostics use the first 100. The auxiliary classifier obtains {100*dg['classifier_validation_accuracy']:.2f}% validation and {100*dg['classifier_test_accuracy']:.2f}% official-test accuracy on real grayscale MNIST.",'CaptionCustom')
    p(f"The main GAN receives predictions in {sum(v>0 for v in dg['class_counts'])} of 10 classes, with {dg['effective_classes']:.2f} effective classes. This assesses class-level coverage on a finite sample; it cannot exclude repeated writing styles within a digit class. The no-smoothing ablation yields {dn['effective_classes']:.2f} effective classes and mean confidence {dn['mean_max_class_probability']:.3f}, compared with {dg['mean_max_class_probability']:.3f} for smoothing. These observations come from one seed per configuration, so I do not infer that smoothing is universally beneficial.")
    p('GAN losses need not fall monotonically because the opponent changes throughout training. Oscillations are therefore interpreted alongside the fixed-latent progress images and diversity histogram. A finite loss without class or visual diversity would not count as successful training. Both ablation sample grids and their complete histories are included in the shared results folder.')
    p('I do not observe complete collapse to one digit: the grid contains different digit shapes and all ten predicted classes occur. However, malformed loops, broken strokes and ambiguous digits remain visible. Class-level coverage alone cannot rule out partial or within-class mode collapse.')

    page();heading('9. Longer training: a follow-up experiment')
    ml=metrics['made_long'];gl=metrics['gan_long'];dgl=diagnostics['gan_long']
    p('The 30-epoch MADE validation curves were still improving near the budget limit. I therefore ran the same two-layer architecture from the same initialization for 120 epochs and also extended the smoothed GAN to 100 epochs. These are follow-up budget experiments, separate from the equal-budget ablations above. AR checkpoint selection still uses validation NLL; the longer GAN uses its predefined final epoch.')
    table([['Model / budget','Validation NLL','Test NLL','Train wall (s)'],['MADE / 30 epochs',f"{m['validation_nll']:.2f}",f"{m['test_nll']:.2f}",f"{m['training_seconds']:.1f}"],['MADE / 120 epochs',f"{ml['validation_nll']:.2f}",f"{ml['test_nll']:.2f}",f"{ml['training_seconds']:.1f}"],['GAN / 30 epochs','Not defined','Not defined',f"{g['training_seconds']:.1f}"],['GAN / 100 epochs','Not defined','Not defined',f"{gl['training_seconds']:.1f}"]],[173,108,108,110])
    p(f"Table 4. The longer MADE checkpoint is selected at epoch {ml['best_epoch']}. Its test NLL changes by {ml['test_nll']-m['test_nll']:+.2f} nats/image relative to the 30-epoch model. Its stored architecture and preprocessing are unchanged; additional training does not guarantee a better result than the linear baseline.",'CaptionCustom')
    t=Table([[picture(RES/'made_long'/'samples.png',231),picture(RES/'gan_long'/'samples.png',231)]],colWidths=[249.5,249.5]);story.append(t)
    p('Figure 6. First 100 uncurated outputs after longer training: two-layer MADE (left) and GAN (right), with the same seed-1234 sampling inputs as the shorter runs. Complete progress grids and epoch histories are included in their result folders.','CaptionCustom')
    p(f"The 100-epoch GAN has {dgl['effective_classes']:.2f} effective predicted classes, mean classifier confidence {dgl['mean_max_class_probability']:.3f}, and confidence >= 0.9 in {100*dgl['fraction_confidence_ge_0_9']:.1f}% of 2,000 generated images. The 30-epoch values are {dg['effective_classes']:.2f}, {dg['mean_max_class_probability']:.3f} and {100*dg['fraction_confidence_ge_0_9']:.1f}%. These quantify changes in the diagnostic classifier's response, not a guaranteed improvement in human-perceived realism.")

    page();heading('10. Discussion and conclusion')
    neighbor=picture(RES/'gan'/'nearest_neighbors.png',102)
    discussion=[Paragraph('Generated / nearest training',styles['CaptionCustom']),neighbor]
    text=[]
    def dp(s):text.append(Paragraph(s,styles['BodyCustom']))
    dp('<b>Likelihood and visual quality.</b> FVSBN provides a compact conceptual baseline and exact likelihood, but its linear logits limit conditional interactions. MADE increases nonlinear capacity while preserving a valid autoregressive distribution. A GAN learns a different objective and retains grayscale detail, so I avoid treating its sharper or softer appearance as evidence of better binary density estimation.')
    dp(f"<b>Sampling cost.</b> The GAN takes {g['inference']['seconds_median']:.4f} s for 100 images, compared with {m['inference']['seconds_median']:.4f} s for this MADE implementation. The difference is consistent with one-pass GAN sampling versus 784 sequential AR steps. This is an implementation-specific benchmark; caching hidden computations could speed up neural autoregressive sampling.")
    dp(f"<b>Memorization check.</b> The first 100 GAN outputs have a minimum nearest-training RMSE of {dg['nearest_train_rmse_min']:.4f}. The adjacent pairs show the first ten outputs and their closest training images. This checks for close pixel-space copies in this sample, but does not prove absence of memorization or measure semantic originality.")
    dp('<b>Limitations.</b> The ablations use one training seed, a limited hyperparameter grid and fixed training budgets. Test-image standard errors capture sampling variability across test images only. Fixed thresholding discards antialiasing and differs from stochastic-binarized MNIST benchmarks, so published NLL figures on other preprocessing should not be compared directly. Classifier metrics may overrate unrealistic digits.')
    dp('<b>Next experiment.</b> I would repeat configurations across seeds, match optimizer steps in the data-size study, and compare equally sized one- and two-layer MADE models. For GANs, I would inspect per-class variation and add a feature-space two-sample evaluation with a validated MNIST embedding, rather than relying on classifier confidence alone.')
    dp('<b>Conclusion.</b> These experiments demonstrate the three required generation mechanisms and their main trade-offs. Explicit autoregressive likelihood supports quantitative density comparison; neural conditionals change the capacity and optimization trade-off; adversarial generation produces grayscale images efficiently but requires multiple checks of stability and diversity. Reproducible evidence is more informative than selecting only the best-looking digits.')
    t=Table([[discussion,text]],colWidths=[130,369]);t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),10)]));story.append(t)
    p('Figure 7. Each row contains a generated sample from the 30-epoch GAN (left) and its closest image among all 55,000 training images (right), using pixel RMSE. No nearest-neighbor result is used to select GAN checkpoints.','CaptionCustom')

    page();heading('Appendix A. Reproduction and evidence')
    url=meta['shared_code_data_url'].strip()
    if url:
        p('Code, data, checkpoints and logs: <link href="'+escape(url,{'"':'&quot;'})+'" color="#165b94">'+escape(url)+'</link>')
    else:
        p('<b>Submission prerequisite:</b> the code/data sharing URL has not yet been supplied. Upload the accompanying submission folder to an accessible shared location, set shared_code_data_url in submission_metadata.json, and rebuild this PDF before submitting. The current local report is not yet submission-ready.')
    p('The accompanying folder contains models.py, train.py, test_models.py, build_report.py, requirements.txt, README.txt, run_colab.ipynb, raw MNIST data, split_indices.npz, and results for every configuration. Each result contains a checkpoint, per-epoch history, measured metrics and sample/progress images. Per-image AR test losses permit paired comparisons. The auxiliary classifier checkpoint and GAN diagnostics are also supplied.')
    sub('A.1 Google Colab / local commands')
    p('1. Select a GPU runtime in Colab and upload/extract the folder, or create a local Python 3.12 environment.<br/>2. Change into the submission folder.<br/>3. Install dependencies and run the checks, experiments and report builder:')
    p('<font name="Courier">pip install -r requirements.txt<br/>python test_models.py<br/>python train.py --stage smoke<br/>python train.py --stage all<br/>python build_report.py</font>','EquationCustom')
    p('The notebook run_colab.ipynb provides the same sequence. Stage selectors ar, gan and evaluate allow separate runs. A completed experiment is skipped when its metrics.json exists; use a fresh folder without results for a clean rerun. The data loader verifies/downloads MNIST using torchvision and reconstructs the seeded split. CPU fallback is supported but slower. Do not change budgets while silently reusing previous result folders.')
    total=sum(r['training_seconds'] for r in metrics.values())
    p(f"The nine generative runs took {total/60:.1f} minutes of measured training wall time in total on the reported GPU, excluding setup, downloads, timing benchmarks and classifier diagnostics. Exact times and outputs vary across devices. The code sets seeds but does not guarantee bitwise-identical CUDA results across platforms.")
    sub('A.2 Evidence map')
    table([['Requirement','Evidence'],['Three tasks / preprocessing / modules','Sections 1-4; models.py and train.py'],['Generated images and training progress','Figures 2-4 and 6; each results subfolder'],['Hyperparameters and ablations','Tables 1, 3, 4 and Figures 1, 5'],['Training and inference runtime','Table 2; metrics.json, including update-only time'],['Code clarity and runnable instructions','README.txt, notebook, comments and mathematical tests'],['Discussion / stability / limitations','Sections 8-10 and nearest-neighbor inspection']], [223,276])
    p('Before submission, confirm that the shared link is accessible to the marker and includes both code and data. Submit 60434764.pdf on Canvas. Preserve the submitted code/data without subsequent edits, as required by the assignment.','SmallCustom')

    page();heading('References')
    p('[1] Jianyuan Guo. CS5494 Lecture 2: Basic of Probability Distributions. Course slides supplied with the assignment. Relevant material: chain rule, Bayesian networks, logistic regression and neural conditional models.')
    p('[2] Jianyuan Guo. CS5494 Lecture 3. Course slides supplied with the assignment. Relevant material: autoregressive factorization, FVSBN (slide 11), neural autoregressive density estimation (slides 13-16).')
    p('[3] Mathieu Germain, Karol Gregor, Iain Murray and Hugo Larochelle. MADE: Masked Autoencoder for Distribution Estimation. ICML, 2015. <link href="https://proceedings.mlr.press/v37/germain15.html" color="#165b94">https://proceedings.mlr.press/v37/germain15.html</link>')
    p('[4] Ian Goodfellow et al. Generative Adversarial Nets. NeurIPS, 2014. <link href="https://arxiv.org/abs/1406.2661" color="#165b94">https://arxiv.org/abs/1406.2661</link>')
    p('[5] PyTorch contributors. torchvision.datasets.MNIST documentation and dataset downloader. <link href="https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html" color="#165b94">https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html</link>')
    sub('Implementation provenance')
    p('The accompanying scripts implement the models with PyTorch layers and optimizers. All reported generative checkpoints are trained from scratch on MNIST; no pretrained generator or external synthetic images are used. The report tables and plots are built from saved run outputs. Course definitions and the cited papers inform the methods; the diagnostic classifier is trained separately and does not supply gradients to any generative model.')
    p('Development assistance: AI assistance was used to develop and review code and report text. Numerical results and sample figures are produced by executing the supplied experiments. The student should review the implementation, findings and applicable course disclosure requirements before submission.','SmallCustom')

    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#ccd7e0'));canvas.line(48,43,A4[0]-48,43)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536273'))
        canvas.drawString(48,29,'CS5494 | Huang ChuanBin | 60434764');canvas.drawRightString(A4[0]-48,29,str(doc.page));canvas.restoreState()
    target=ROOT/'60434764.pdf'
    doc=SimpleDocTemplate(str(target),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=56,
        title='CS5494 Assignment 1 - Huang ChuanBin - 60434764',author=meta['name'])
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(target)


if __name__=='__main__':main()
