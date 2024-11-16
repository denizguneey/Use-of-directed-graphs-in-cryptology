import numpy as np
from sympy import Matrix, pprint,oo
from collections import Counter
from enum import Enum
class AppType(Enum):
    Uzaklık = 1
    Komsuluk = 2
    Laplasyan = 3
alphabet = 'ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ @'
#______________________________INPUTS_________________________________________
#nodes= ['1', '2', '3', '4'] #uzaklık 1
#nodes= ['1', '2', '3', '4', '5'] #uzaklık 2
#nodes= ['1', '2', '3', '4', '5'] #komşuluk 3
#nodes= ['1', '2', '3', '4', '5','6'] #komşuluk 4
nodes= ['1', '2', '3', '4', '5','6','7'] #laplasyan 5
#edges= ['1-2', '1-3', '3-1', '3-2','4-1'] #uzaklık 1 
#edges= ['1-2', '2-3', '3-1', '3-5','4-2','4-5', '5-4'] #uzaklık 2 
#edges= ['1-2', '2-3', '2-4', '3-5','4-1','5-3', '5-4'] #komşuluk 3 
#edges= ['1-2', '2-3', '3-4', '4-5','5-4','5-6', '6-1'] #komşuluk 4 
edges= ['1-3', '2-1', '4-2', '4-3','5-4','6-4', '7-5','7-6'] #laplasyan 5 
type= AppType.Laplasyan
#_______________________________________________________________________
n = len(nodes)
plain_text ="MATEMATİK EVRENİN DİLİDİR"
node_indices = {node: index for index, node in enumerate(nodes)}
edges_indices = [(node_indices[edge[0]], node_indices[edge[2]]) for edge in edges]
matrix = Matrix([[oo] * n for _ in range(n)])
for i in range(n):
    matrix[i, i] = 0
for edge in edges_indices:
    matrix[edge[0], edge[1]] = 1
distance_matrix = matrix.copy()
for k in range(n):
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = min(distance_matrix[i, j], distance_matrix[i, k] + distance_matrix[k, j])
for i in range(n):
    for j in range(n):
        if matrix[i, j] == oo:
            matrix[i, j] = 0
        if distance_matrix[i, j] == oo:
            distance_matrix[i, j] = 0
print ("\nDüz metin:", plain_text)
if type == AppType.Komsuluk:
    print ("\n Komşuluk matrisi A(G):")
    print("\n")
    pprint(matrix)    
    try:   
        matrix_inverse = matrix.inv_mod( len(alphabet)) 
    except:    
        unit_matrixx = Matrix.eye(n)*2
        print("\ndet(A(G))=0")
        print ("\nK(G)=A(G)+2I")
        matrix=matrix+unit_matrixx
        print ("\nAnahtar matrisi K(G):")
        print("\n")
        pprint(matrix)
        det_inv=pow(matrix.det(),-1,len(alphabet)) 
        matrix_inverse = matrix.inv_mod( len(alphabet)) 
if type == AppType.Uzaklık:
    print ("\nUzaklık matrisi D(G):")
    print("\n")
    pprint(distance_matrix)
    try:
        
        matrix_inverse = distance_matrix.inv_mod( len(alphabet))
        matrix=distance_matrix 
    except:
        unit_matrixx = Matrix.eye(n)*2
        print("\ndet(D(G))=0")
        print ("\nK(G)=D(G)+2I")
        matrix=distance_matrix+unit_matrixx
        print ("\nAnahtar matrisi K(G):")
        print("\n")
        pprint(matrix)
        det_inv=pow(matrix.det(),-1,len(alphabet)) 
        matrix_inverse = matrix.inv_mod( len(alphabet))
if type == AppType.Laplasyan:
    points = [int(edge[i]) for edge in edges for i in [0, 2]]
    count = Counter(points)
    degree_matrix = Matrix([[0] * n for _ in range(n)])
    for item in count.items():
        degree_matrix[item[0] - 1, item[0] - 1] = item[1]
    print ("Derece matrisi D(G):")
    print("\n")
    pprint (degree_matrix)
    print("\n")
    print ("Komşuluk matrisi A(G):")
    print("\n")
    pprint (matrix)
    
    print ("\nLaplasyan matrisi D(G)-A(G)= L(G):")
    print("\n")
    matrix= degree_matrix - matrix
    pprint(matrix)
    matrix_inverse = matrix.inv_mod( len(alphabet)) 
block_size = matrix.shape[0]
def text_to_numbers(text):
    return [alphabet.index(char) for char in text if char in alphabet]
def pad_text(text, block_size):
    padding_length = (block_size - len(text) % block_size) % block_size
    return text + ' ' * padding_length
def split_into_blocks(text, block_size):
    return [text[i:i+block_size] for i in range(0, len(text), block_size)]
padded_text = pad_text(plain_text, block_size)
text_blocks = split_into_blocks(padded_text, block_size)
number_blocks = [text_to_numbers(block) for block in text_blocks]
def encrypt_decrypt_block(block, key):
    vector = np.array(block)
    encrypted_vector = np.dot(vector, key) % len(alphabet)
    return encrypted_vector.astype(int).tolist()
encrypted_blocks = [encrypt_decrypt_block(block, matrix) for block in number_blocks]
encrypted_numbers = [num for block in encrypted_blocks for num in block]
encrypted_text = ''.join([alphabet[num] for num in encrypted_numbers])
print("\nMetin Blokları:")
for i, block in enumerate(text_blocks):
    print(f"{block} : {number_blocks[i]}")
print("\nŞifreli metin:", encrypted_text)    
print ("\nAnahtar matrisi tersi K(G)^(-1):")
print("\n")
pprint (matrix_inverse)
padded_text = pad_text(encrypted_text, block_size)
text_blocks = split_into_blocks(padded_text, block_size)
number_blocks = [text_to_numbers(block) for block in text_blocks]
print("\nŞifreli Metin Blokları:")
for i, block in enumerate(text_blocks):
    print(f"{block} : {number_blocks[i]}")
decrypted_blocks = [encrypt_decrypt_block(block, matrix_inverse) for block in number_blocks]
decrypted_numbers = [num for block in decrypted_blocks for num in block]
decrypted_text = ''.join([alphabet[num] for num in decrypted_numbers])
print ("\nDüz Metin:", decrypted_text)


