import os
import pathlib
import gzip

from openmm import XmlSerializer


def serialize_and_compress(item):
    """Serialize an OpenMM System, State, or Integrator and compress.

    Parameters
    ----------
    item : System, State, or Integrator
        The OpenMM object to serialize and compress.

    Returns
    -------
    bytes : bytes
        The compressed serialized OpenMM object.
    """
    serialized = XmlSerializer.serialize(item).encode()
    data = gzip.compress(serialized)
    return data


def decompress_and_deserialize(data: bytes):
    """Recover an OpenMM object from compression.

    Parameters
    ----------
    data : bytes
        Bytes containing a gzip compressed XML serialization
        of an OpenMM object.

    Returns
    -------
    deserialized
        The deserialized OpenMM object.
    """
    decompressed = gzip.decompress(data).decode()
    deserialized = XmlSerializer.deserialize(decompressed)
    return deserialized


def serialize(item, filename: pathlib.Path):
    """
    Serialize an OpenMM System, State, or Integrator.

    Parameters
    ----------
    item : System, State, or Integrator
        The thing to be serialized
    filename : str
        The filename to serialize to
    """

    # Create parent directory if it doesn't exist
    filename_basedir = filename.parent
    if not filename_basedir.exists():
        os.makedirs(filename_basedir)

    if filename.suffix == ".gz":
        import gzip

        with gzip.open(filename, mode="wb") as outfile:
            serialized_thing = XmlSerializer.serialize(item)
            outfile.write(serialized_thing.encode())
    if filename.suffix == ".bz2":
        import bz2

        with bz2.open(filename, mode="wb") as outfile:
            serialized_thing = XmlSerializer.serialize(item)
            outfile.write(serialized_thing.encode())
    else:
        with open(filename, mode="w") as outfile:
            serialized_thing = XmlSerializer.serialize(item)
            outfile.write(serialized_thing)


def deserialize(filename: pathlib.Path):
    """
    Deserialize an OpenMM System, State, or Integrator.

    Parameters
    ----------
    item : System, State, or Integrator
        The thing to be serialized
    filename : str
        The filename to serialize to
    """
    from openmm import XmlSerializer

    # Create parent directory if it doesn't exist
    filename_basedir = filename.parent
    if not filename_basedir.exists():
        os.makedirs(filename_basedir)

    if filename.suffix == ".gz":
        import gzip

        with gzip.open(filename, mode="rb") as infile:
            serialized_thing = infile.read().decode()
            item = XmlSerializer.deserialize(serialized_thing)
    if filename.suffix == ".bz2":
        import bz2

        with bz2.open(filename, mode="rb") as infile:
            serialized_thing = infile.read().decode()
            item = XmlSerializer.deserialize(serialized_thing)
    else:
        with open(filename) as infile:
            serialized_thing = infile.read()
            item = XmlSerializer.deserialize(serialized_thing)

    return item
