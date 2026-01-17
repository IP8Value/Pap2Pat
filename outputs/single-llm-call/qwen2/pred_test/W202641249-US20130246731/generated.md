# DESCRIPTION

## BACKGROUND

Graph-processing applications are rapidly becoming essential in various domains, including social networks, road traffic management, and biological networks, where data naturally exhibit a graph structure. These applications often require middleware support for efficient data storage and accelerated graph queries. Modern graph applications demand high-volume storage, low-latency updates, and interactive responsiveness. Individually, these requirements are not unique, but collectively, they pose significant challenges to both state-of-the-art storage and data-processing systems.

One of the primary challenges is achieving scalability and consistency. Many graph applications operate on the critical path of online workflows, necessitating a combination of adequate query throughput and data ingestion rate. For instance, Facebook processes over 200,000 events per second, while Twitter ingests approximately 80 terabytes of new data daily. Without robust consistency semantics, such as transactional guarantees, developing distributed graph applications becomes difficult and error-prone.

Another challenge is event-driven processing. Some graph applications, such as those used by emergency technicians to respond to incidents, must be real-time to be effective. For example, the California highway road sensors require the ingestion of new data every 30 seconds for over 26,000 sensors. These applications are largely event-driven, triggering computation to predict the spread and duration of incidents. Supporting an API with flexible event processing on the graph store simplifies the development of these applications and avoids redundant client-side computation on event detection. Events can also be used to monitor and dynamically optimize the store itself, further improving query performance.

However, state-of-the-art solutions are not designed to address these challenges simultaneously. Traditional storage systems like relational databases and NoSQL stores do not inherently retain the structure of the graph and are unsuitable for computing graph algorithms. Distributed in-memory stores, such as GemFire, provide dynamic scalability and high performance while supporting transactions but lack native support for graph objects, making graph queries inefficient. Contemporary data-processing frameworks like MapReduce, Pregel, Spark, or GraphLab optimize for batch analysis by assuming data is largely read-only, making them ill-suited for concurrent read and write queries. Specialized graph databases perform complex graph queries quickly and concurrently with graph updates but are limited to a single machine or a set of replicated images and do not scale with query rate, storage capacity, or data ingestion rate. None of these systems support event-driven processing.

## DESCRIPTION OF EMBODIMENTS

### Notation and Nomenclature

In this patent application, the following terms and notations are used:

- **Graph**: A collection of vertices and edges, where vertices represent entities and edges represent relationships between entities.
- **Vertex**: A node in the graph representing an entity.
- **Edge**: A connection between two vertices representing a relationship.
- **Property**: An attribute attached to a vertex or edge.
- **View**: A subgraph of interest on which applications can run graph algorithms and register event handlers.
- **Event Handler**: A function that is invoked when a specified event occurs in a view.
- **Distributed Transactions**: A mechanism to ensure consistency and concurrency across multiple servers.
- **Mini-Transaction**: A performance-optimized implementation of the two-phase commit protocol provided by Sinfonia.
- **Memnode**: A flat memory region per server in a distributed shared memory system.
- **Global Address Space**: A combined memory region from multiple memnodes.
- **BSP (Bulk Synchronous Parallel)**: A parallel computing model where computation proceeds in a series of supersteps separated by global synchronization points.

### Overview of Discussion

This patent application describes Concerto, a distributed graph store that combines the functionality of specialized graph databases with the ability to scale and support event-driven applications. Concerto provides distributed, in-memory, transactional storage of graph elements and introduces the concept of graph views to simplify application development and improve performance. The system is designed to handle high-volume storage, low-latency updates, and interactive responsiveness, making it suitable for a wide range of graph-processing applications.

### Example Graph Storage System

Concerto stores graph objects in memory and across distributed commodity servers in data centers. A distributed shared memory implementation provides a global address space on which graph objects are allocated. Graph traversals take place on the distributed graph representation using server-side RPC calls batched for performance. Concerto's key contributions are in the in-memory graph representation and the use of efficient, distributed transactions to provide concurrent access and online data migration.

### Example Graph Structure

Concerto has three basic data types to store the application graph data: vertex, edge, and property. A property element contains attributes and can be attached to a vertex or edge. Vertices and edges can have multiple properties. Concerto exposes APIs to graph applications to create and update the above graph elements. New graph objects are allocated on a global address space provided by Sinfonia, a distributed shared memory system. Sinfonia exposes a flat memory region per server called a memnode, which are combined to create a single global address space.

Concerto stores the logical graph using a layout optimized for in-memory reads and inserts. Vertices, edges, and properties are represented as records with pointers. A vertex has a pointer to a list of its outgoing edges. An edge has pointers to its source and destination vertices and to the next edge of the source vertex. Thus, all outgoing edges of a vertex can be accessed consecutively starting from the first edge. Co-locating vertices and edges in contiguous blocks of memory and storing pointers to related graph objects allow graph traversals to be performed quickly at the cost of additional storage. Properties are chained together as a list, and both vertex and edge records point to the head of their property lists.

Each vertex and edge is a fixed-size record, while properties can be of variable size. Using an appropriate fixed size, a vertex or edge can be retrieved in one read transaction (one network roundtrip between a client and a Concerto server) as both the address and size of the data are known in advance. However, accessing properties of a vertex or edge may require more than one transaction. First, the vertex has to be read to determine the address of the property, and then the property is read in the next transaction. In some applications, certain properties are accessed often. To retrieve these frequently accessed objects in one read transaction, properties can optionally be embedded in the vertex or edge records.

### Example Distributed Storage and Memory Allocation

Concerto uses distributed transactions to provide consistency and concurrency for graph allocation, access, and updates. Unlike simple key-value data, graph data can seldom be partitioned into shared-nothing regions, and hence support for transactions that occur across machines is necessary. To balance consistency with efficiency, Concerto leverages a distributed compare-and-swap primitive called a mini-transaction provided by Sinfonia to support such distributed transactions. Mini-transactions are a performance-optimized implementation of the two-phase commit protocol. Concerto also provides other optimizations to minimize the number of transactions used, including batching graph operations during traversals and reducing the number of indirections for graph object access. Using these optimizations, Concerto can, in the common case, perform reads of vertices, edges, or attributes in a single network roundtrip and finish writes in two network roundtrips. By comparison, transactionally updating even a single value in GemFire requires at least three network roundtrips.

During allocation of new graph elements (e.g., vertex, edge), it is important to ensure a unique address is assigned to the graph element even if two concurrent users request memory. Concerto uses transactions to achieve this. Whenever an allocation request is received, the Concerto graph allocator contacts the Sinfonia memnode. Upon allocation of an address space, an entry is made to the allocation metadata on the memnode. Concerto wraps these operations in transactions, ensuring that the metadata for the allocator remains consistent during concurrent allocation requests. Note that the use of transactions to allocate and manage each element incurs overhead, especially for vertices and edges, which are only a few tens of bytes. To reduce this, Concerto pre-allocates large memory blocks from memnodes and appends new vertices and edges until the block fills up. Pre-allocated blocks reduce the amount of metadata stored on memnodes and the number of network roundtrips (and possible write conflicts) from allocation requests.

### Example Online Data Migration

Concerto uses transactions to provide online data migration for an application to optimize a graph partition. This can be used, for example, when adding or removing servers or when handling data hotspots. Table 2 shows the three migrate functions available to applications. These functions implement migration as a series of tasks wrapped inside distributed transactions. For example, when migrating a vertex, the vertex and its associated data are copied to the new server, the original copy is deleted, and all incoming pointers to the vertex are updated. These tasks happen inside a transaction during which time other non-conflicting operations can continue concurrently.

### Example Fault Toleration Structure

Concerto simplifies the graph store architecture by delegating most of the fault recovery mechanisms to Sinfonia. Sinfonia provides atomicity, consistency, isolation, durability (ACID), and availability if replication is enabled. These guarantees are independent of client failures and the size of the graph. This design choice ensures that the graph store can easily be ported to other platforms such as distributed key-value stores. The Concerto prototype uses Sinfonia's disk-logging mechanisms to recover from memnode failures.

Sinfonia's fault-tolerant global address space implies that data stored in Concerto is recoverable. However, mechanisms in Concerto are needed to regain consistency (upon recovery) of the graph allocators. Graph allocators store all their metadata in the memnodes. If a graph allocator fails, some of the memory may be leaked (e.g., pre-allocated blocks may be left dangling). The recovery process in Concerto goes through the allocator metadata in each memnode and entrusts any dangling memory block to an active graph allocator.

Unlike data operations, event processing in Concerto's current prototype is not completely fault-tolerant. The difference in guarantees occurs because events are processed asynchronously to isolate query performance from event processing. As a result, an untimely fault can result in lost events. For example, when an update occurs, the write operation may return results to the client even though the triggered event processing code may still be executing a computation. If there is a fault before the update operation returns, then Concerto's recovery process will ensure that both the write operation and the event-processing code is correctly re-executed (or the client is notified of the failure and can retry). However, if a fault occurs after the update operation completes but before the event processing code completes, then the event may be lost. One can make event processing fault-tolerant by using a fault-tolerant message queue, which is planned as future work.

### Example Computer System

Concerto consists of approximately 4,500 lines of C++ code for distributed data allocation, query API, distributed graph traversals, and event processing. These lines of code do not include the Sinfonia codebase. Concerto can be deployed on a cluster of commodity servers, each equipped with multi-core processors, ample RAM, and high-speed networking. The system is designed to run in-memory, leveraging the distributed shared memory provided by Sinfonia.

### Example Method of Use

#### Graph Representation and Allocation

1. **Graph Representation**: Concerto stores graph objects using a layout optimized for in-memory reads and inserts. Vertices, edges, and properties are represented as records with pointers. Vertices have pointers to their outgoing edges, and edges have pointers to their source and destination vertices and to the next edge of the source vertex. Properties are chained together as a list, and both vertex and edge records point to the head of their property lists.
2. **Graph Allocation**: Concerto uses transactions to ensure unique addresses are assigned to new graph elements. The graph allocator contacts the Sinfonia memnode to allocate an address space and updates the allocation metadata. Concerto pre-allocates large memory blocks to reduce metadata storage and network roundtrips.

#### Graph Updates and Transactions

1. **Graph Updates**: Transactions are used to allow in-place updates to existing graph elements while maintaining consistency and concurrency. The Concerto transaction API calls the Sinfonia mini-transaction subsystem to update graph elements on distributed machines.
2. **Online Data Migration**: Concerto uses transactions to provide online data migration for optimizing graph partitions. Functions for migrating vertices, edges, and properties are implemented as a series of tasks wrapped inside distributed transactions.

#### Graph Views and Event Processing

1. **Graph Views**: Concerto introduces the concept of graph views, which are subgraphs of interest on which applications can run graph algorithms and register event handlers. Views are created using the View class and can be composed using set operations such as union, intersection, and subtraction.
2. **Event Processing**: Applications can register event handlers with views to process events such as read and write operations. The View API provides functions that are invoked when specified events occur in the view. Event processing is handled asynchronously to isolate query performance from event processing.

#### Fault Tolerance and Security

1. **Fault Tolerance**: Concerto delegates most fault recovery mechanisms to Sinfonia, which provides ACID guarantees and availability. The recovery process in Concerto ensures that the graph allocators regain consistency upon recovery.
2. **Security**: Concerto assumes that functions registered with views are written by trusted applications. Future work includes implementing security features such as sandboxes to limit the power of these functions.

#### Performance and Scalability

1. **Performance**: Concerto is designed to handle high-volume storage, low-latency updates, and interactive responsiveness. It outperforms other systems in insertion throughput, query latency, and memory footprint.
2. **Scalability**: Concerto can leverage distributed parallelism to improve performance. The system can dynamically load balance data to mitigate workload hotspots and optimize query performance.

By combining distributed, in-memory, transactional storage with the concept of graph views and event processing, Concerto provides a powerful and flexible solution for a wide range of graph-processing applications.