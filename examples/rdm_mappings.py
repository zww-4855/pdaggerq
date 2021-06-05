"""
Example for vacuum normal ordering the T2 operator for 2-RDM theory
"""
import pdaggerq

def main():
    print("T2 mappings")
    # need cumulant decomposition on 3-RDM terms
    # to simplify to a 2-RDM + 1-RDM expression
    ahat = pdaggerq.pq_helper('true')

    ahat.set_string(['i*','j*','k','n*','m', 'l'])
    ahat.add_new_string()
    ahat.set_string(['i*','j','k', 'n*','m*', 'l'])
    ahat.add_new_string()

    ahat.simplify()
    ahat.print()

    ahat.clear()

    print("Q -> D")
    ahat = pdaggerq.pq_helper('true')

    ahat.set_string(['i', 'j', 'k*', 'l*'])
    ahat.add_new_string()

    ahat.simplify()
    ahat.print()

    ahat.clear()

    print("G -> D")
    ahat = pdaggerq.pq_helper('true')
    ahat.set_string(['i*', 'j', 'k*', 'l'])
    ahat.add_new_string()
    ahat.simplify()
    ahat.print()
    ahat.clear()

if __name__ == "__main__":
    main()