def dna_to_complement(dna):
    # Create the complementary DNA
    complement = ''
    for nucleotide in dna:
        if nucleotide == 'A':
            complement += 'T'
        elif nucleotide == 'T':
            complement += 'A'
        elif nucleotide == 'C':
            complement += 'G'
        elif nucleotide == 'G':
            complement += 'C'
    return complement

def dna_to_mrna(dna):
    # Convert DNA to mRNA (replace T with U)
    return dna.replace('T', 'U')

def mrna_to_amino_acids(mrna):
    # Codon to amino acid mapping
    codon_table = {
        'AUG': 'Met (M)', 'UUU': 'Phe (F)', 'UUC': 'Phe (F)', 'UUA': 'Leu (L)', 'UUG': 'Leu (L)',
        'CUU': 'Leu (L)', 'CUC': 'Leu (L)', 'CUA': 'Leu (L)', 'CUG': 'Leu (L)', 'AUU': 'Ile (I)',
        'AUC': 'Ile (I)', 'AUA': 'Ile (I)', 'GUU': 'Val (V)', 'GUC': 'Val (V)', 'GUA': 'Val (V)',
        'GUG': 'Val (V)', 'UCU': 'Ser (S)', 'UCC': 'Ser (S)', 'UCA': 'Ser (S)', 'UCG': 'Ser (S)',
        'CCU': 'Pro (P)', 'CCC': 'Pro (P)', 'CCA': 'Pro (P)', 'CCG': 'Pro (P)', 'ACU': 'Thr (T)',
        'ACC': 'Thr (T)', 'ACA': 'Thr (T)', 'ACG': 'Thr (T)', 'GCU': 'Ala (A)', 'GCC': 'Ala (A)',
        'GCA': 'Ala (A)', 'GCG': 'Ala (A)', 'UAU': 'Tyr (Y)', 'UAC': 'Tyr (Y)', 'UAA': 'Stop',
        'UAG': 'Stop', 'CAU': 'His (H)', 'CAC': 'His (H)', 'CAA': 'Gln (Q)', 'CAG': 'Gln (Q)',
        'AAU': 'Asn (N)', 'AAC': 'Asn (N)', 'AAA': 'Lys (K)', 'AAG': 'Lys (K)', 'GAU': 'Asp (D)',
        'GAC': 'Asp (D)', 'GAA': 'Glu (E)', 'GAG': 'Glu (E)', 'UGU': 'Cys (C)', 'UGC': 'Cys (C)',
        'UGA': 'Stop', 'UGG': 'Trp (W)', 'CGU': 'Arg (R)', 'CGC': 'Arg (R)', 'CGA': 'Arg (R)',
        'CGG': 'Arg (R)', 'GGU': 'Gly (G)', 'GGC': 'Gly (G)', 'GGA': 'Gly (G)', 'GGG': 'Gly (G)',
    }

    amino_acids = []
    for i in range(0, len(mrna), 3):
        codon = mrna[i:i+3]
        if codon in codon_table:
            amino_acids.append(codon_table[codon])
        else:
            amino_acids.append('Unknown')

    return ' - '.join(amino_acids)

def translate_dna(dna):
    # Check if the length of DNA is a multiple of 3
    if len(dna) % 3 != 0:
        return "Error: Input DNA sequence length must be a multiple of 3."

    # Step 1: Get the complementary DNA
    complement = dna_to_complement(dna)

    # Step 2: Convert DNA to mRNA
    mrna = dna_to_mrna(dna)

    # Step 3: Convert mRNA to amino acid sequence
    amino_acid_sequence = mrna_to_amino_acids(mrna)

    return (f"Input DNA: {dna}\n"
            f"Complement: {complement}\n"
            f"mRNA: {mrna}\n"
            f"Amino Acid Sequence: {amino_acid_sequence}")

# Allow user input for DNA sequence with specific prompt
input_dna = input("Input DNA = ").upper()

# Validate input
if input_dna.isalpha() and all(nucleotide in 'ATCG' for nucleotide in input_dna):
    result = translate_dna(input_dna)
    print(result)
else:
    print("Invalid input! Please enter a valid DNA sequence consisting of A, T, C, and G.")
