
import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# energy equation

pq.set_left_operators([['1']])

print('')
print('    < 0 | e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t2'])
pq.add_st_operator(1.0,['v'],['t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
energy_terms = pq.fully_contracted_strings()
print('printing terms in energy expression:')
for my_term in energy_terms:
    print(my_term)

pq.clear()


# doubles equations

pq.set_left_operators([['e2(i,j,b,a)']])

print('')
print('    < 0 | i* j* b a e(-T) H e(T) | 0> :')
print('')


pq.add_st_operator(1.0,['f'],['t2'])
pq.add_st_operator(1.0,['v'],['t2'])
#pq.add_triple_commutator(1.0/6.0,['v'],['v'],['t2'],['t2'])
#pq.add_triple_commutator(1.0/6.0,['v','v','t2','t2'])
#pq.add_operator_product( 1.0, ['v','t4'])
#pq.add_st_operator(1.0,['v'],['t4'])
pq.simplify()

# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term)

doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
for my_term in doubles_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='doubles_res',
                                output_variables=('a', 'b', 'i', 'j')))
    print()

pq.clear()
import sys
sys.exit()
# quad equations

pq.set_left_operators([['e4(i,j,k,l,d,c,b,a)']])

print('')
print('    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :')
print('')

#pq.add_st_operator(1.0,['f'],['t4'])
pq.add_double_commutator(0.5,['v'],['t2'],['t2'])
#pq.add_st_operator(1.0,['v'],['t2'])
pq.simplify()
# grab list of fully-contracted strings, then print
triples_residual_terms = pq.fully_contracted_strings()
for my_term in triples_residual_terms:
    print(my_term)

triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
for my_term in triples_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='quads_res',
                                output_variables=('a', 'b', 'c','d', 'i', 'j', 'k','l')))
    print()

pq.clear()


# E(t)

pq.set_left_operators([['l2']])

print('')
print('    E(t)')
print('')

pq.add_commutator(1.0,['v'],['t4'])
pq.simplify()

# grab list of fully-contracted strings, then print
e_t_terms = pq.fully_contracted_strings()
for my_term in e_t_terms:
    print(my_term)

e_t_terms = contracted_strings_to_tensor_terms(e_t_terms)
for my_term in e_t_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='energy'))
    print()

pq.clear()

