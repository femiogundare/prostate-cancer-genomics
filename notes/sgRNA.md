### Introduction to CRISPR-Cas9 Technology
The popularity of CRISPR is largely due to its simplicity. The CRISPR-Cas system relies on two main components: a **guide RNA (gRNA)** and **CRISPR-associated (Cas) nuclease**
* The guide RNA is a specific RNA sequence that recognizes the target DNA region of interest and directs the Cas nuclease there for editing. The gRNA is made up of two parts: **crispr RNA (crRNA)**, a 17-20 nucleotide sequence complementary to the target DNA, and a **tracr RNA**, which serves as a binding scaffold for the Cas nuclease.
* The CRISPR-associated protein is a non-specific endonuclease. It is directed to the specific DNA locus by a gRNA, where it makes a double-strand break. There are several versions of Cas nucleases isolated from different bacteria. The most commonly used one is the Cas9 nuclease from Streptococcus pyogenes.

### Difference Between gRNA and sgRNA
<p>sgRNA is an abbreviation for “single guide RNA.” As the name implies, sgRNA is a single RNA molecule that contains both the custom-designed short crRNA sequence **fused** to the scaffold tracrRNA sequence. sgRNA can be synthetically generated or made in vitro or in vivo from a DNA template.</p>
<p>While crRNAs and tracrRNAs exist as two separate RNA molecules in nature, sgRNAs have become the most popular format for CRISPR guide RNAs with researchers, so the sgRNA and gRNA terms are often used with the same meaning in the CRISPR community these days. However, some researchers are still using guide RNAs with the crRNA and tracrRNA components separate, which are commonly referred to as 2-piece gRNAs or simply as cr:tracrRNAs (pronounced CRISPR tracer RNAs).</p>
<p>In a nutshell, gRNA is the term that describes all CRISPR guide RNA formats, and sgRNA refers to the simpler alternative that combines both the crRNA and tracrRNA elements into a single RNA molecule.</p>

### The PAM Sequence and the Cas Nuclease
<p>Each Cas nuclease binds to its target sequence only in presence of a specific sequence, called **protospacer adjacent motif (PAM)**, on the non-targeted DNA strand. Therefore, the locations in the genome that can be targeted by different Cas proteins are limited by the locations of these PAM sequences. The nuclease cuts 3-4 nucleotides upstream of the PAM sequence.</p>
<p>Cas nucleases isolated from different bacterial species recognize different PAM sequences. For instance, the SpCas9 nuclease cuts upstream of the PAM sequence 5′-NGG-3′ (where “N” can be any nucleotide base), while the PAM sequence 5′-NNGRR(N)-3′ is required for SaCas9 (from Staphylococcus aureus) to target a DNA region for editing.</p>
<p>Although the PAM sequence itself is essential for cleavage, it should not be included in the single guide RNA sequence.</p>

### Important considerations and limiting factors for sgRNA design
* The GC content of the sgRNA is important, as higher GC content will make it more stable - it should be 40-80%.
* The length of the sgRNA should be between 17-24 nucleotides, depending on the specific Cas nuclease you’re using. * Shorter sequences can minimize off-target effects, however, if the sequence is too short, the reverse can also occur.
* Mismatches between gRNA and target site can lead to off-target effects, depending on the number of mismatches and their position/s.
* It may be necessary to design multiple sgRNAs for each gene of interest, due to the fact that activity and specificity can be unpredictable.

### Comparison of Different sgRNA Formats
#### A. Plasmid-expressed sgRNA
<p>One of the original methods of making sgRNAs involves expressing the guide RNA sequence in cells from a transfected plasmid. In this method, the sgRNA sequence is cloned into a plasmid vector, which is then introduced into cells. The cells use their normal RNA polymerase enzyme to transcribe the genetic information in the newly introduced DNA to generate the sgRNA.</p>

<p>Cloning the guide RNA plasmid generally requires about 1–2 weeks of lab time prior to the actual CRISPR experiment. The plasmid approach is also more prone to off-target effects than other methods because the guide is expressed over longer periods of time. The plasmid DNA can integrate into the cellular genome, which can result in adverse effects and problematic for downstream applications, and can even cause cell death.</p>

#### B. In Vitro-transcribed sgRNA
<p>Another method for making sgRNA, termed in vitro transcription (IVT), involves transcribing the sgRNA from the corresponding DNA sequence outside the cell. First, a DNA template is designed that contains the guide sequence and an additional RNA polymerase promoter site upstream of the sgRNA sequence. The sgRNA is then transcribed using kits that contain reagents and recombinant RNA polymerase.</p>

<p>Synthesizing sgRNA using the IVT approach requires about 1–3 days. However, the method suffers from certain drawbacks. It is labor-intensive and prone to errors, and the in vitro-transcribed sgRNA generally needs additional purification before it can be used in CRISPR experiments. These limitations result in highly variable editing efficiencies and increased risk of off-target effects.</p>

<p>Another issue associated with IVT sgRNA is that it can have negative effects on certain cell types. Several studies have found that IVT sgRNA can trigger innate immune responses in human and murine cells, causing cytotoxicity and apoptosis. This is thought to be attributed to the 5’ triphosphate group of IVT sgRNAs, which is recognized in the cytoplasm by DDX58, an antiviral innate immune response receptor. DDX58 then produces type I interferons and proinflammatory cytokines, leading to cell death.</p>

#### C. Synthetic sgRNA
<p>The plasmid and IVT methods for sgRNA generation suffer from limitations such as variable editing efficiencies, high off-target effects, and cumbersome implementation. The need for a better alternative-fueled commercial production of synthetic sgRNA.</p>

<p>Synthego generates high-quality synthetic sgRNA using chemical synthesis. Transfecting synthetic sgRNA pre-complexed to the Cas protein in the ribonucleoprotein (RNP) format generated the highest CRISPR editing efficiencies.</p>

<p>While the cost of generating synthetic sgRNAs was initially a barrier for researchers who wanted to improve their CRISPR editing using synthetic sgRNAs, recent technological developments have alleviated those concerns. Synthego’s high-throughput, automated, and scalable RNA synthesis platform enables the production of high fidelity sgRNAs that demonstrate improved CRISPR accessibility across all labs.</p>


### Reference
- [How To Use CRISPR: Your Guide to Successful Genome Engineering](https://www.synthego.com/guide/how-to-use-crispr/sgrna#:~:text=sgRNA%20is%20an%20abbreviation%20for,vivo%20from%20a%20DNA%20template.)