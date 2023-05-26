import streamlit as st
from gqlalchemy import Memgraph

# Connect to Memgraph using GQLAlchemy
memgraph = Memgraph('localhost', 7687)

# Execute a query
query = 'MATCH (n) RETURN n LIMIT 10'
result = memgraph.execute(query)

# Display the result in Streamlit
st.write("Result of Memgraph query:")
if result is not None:
    for record in result:
        st.write(record)
else:
    st.write("No results found.")