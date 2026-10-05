"""CPU archive primitive/rejection checks; no model or native experiment fixture."""
import gzip
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_repeat_local_archive_CPU_tests',HERE/'verify_repeat_delivery_local.py')
V=importlib.util.module_from_spec(spec);sys.modules[spec.name]=V;spec.loader.exec_module(V)
NAME=V.D+'/CPU_primitives.txt'


def header(name=NAME,size=1,kind=tarfile.REGTYPE):
    info=tarfile.TarInfo(name);info.type=kind;info.size=size;info.mode=0o644;info.uid=info.gid=0;info.mtime=0
    return info.tobuf(format=tarfile.USTAR_FORMAT,encoding='ascii',errors='strict')


def raw_tar(name=NAME,data=b'x',kind=tarfile.REGTYPE):
    return header(name,len(data),kind)+data+b'\0'*((-len(data))%512)+b'\0'*1024


class LocalArchiveCPUChecks(unittest.TestCase):
    def parse_archive(self,raw):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'CPU_metadata_only.tar.gz'
            with gzip.open(path,'wb') as stream:stream.write(raw)
            return V.plain_tar(path)

    def test_small_regular_USTAR_stream(self):
        self.assertEqual(self.parse_archive(raw_tar()),[(NAME,b'x')])

    def test_traversal_name_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar(V.D+'/../escape.txt'))

    def test_absolute_name_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar('/absolute.txt'))

    def test_symlink_header_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar(kind=tarfile.SYMTYPE))

    def test_hardlink_header_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar(kind=tarfile.LNKTYPE))

    def test_PAX_extension_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar(kind=tarfile.XHDTYPE))

    def test_GNU_longname_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar(kind=tarfile.GNUTYPE_LONGNAME))

    def test_duplicate_member_rejected(self):
        member=header()+b'x'+b'\0'*511
        with self.assertRaises(ValueError):self.parse_archive(member+member+b'\0'*1024)

    def test_nonzero_padding_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(header()+b'x'+b'z'+b'\0'*510+b'\0'*1024)

    def test_single_terminal_block_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar()[:-512])

    def test_hidden_trailing_member_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar()+header(V.D+'/hidden.txt')+b'x'+b'\0'*511)

    def test_truncated_payload_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(header(size=513)+b'x')

    def test_oversized_member_header_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(header(size=V.MAX_FILE+1)+b'\0'*1024)

    def test_model_binary_payload_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar('models/model.safetensors'))

    def test_SDK_json_payload_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar(V.D+'/runtime-cache/private-sdk/metadata.json'))

    def test_unknown_experiment_scope_rejected(self):
        with self.assertRaises(ValueError):self.parse_archive(raw_tar('experiments/prefix_io_v1/runs/unknown/result.json'))

    def test_duplicate_JSON_key_rejected(self):
        with self.assertRaises(ValueError):V.parse(b'{"a":1,"a":2}')

    def test_nonfinite_JSON_rejected(self):
        with self.assertRaises(ValueError):V.parse(b'{"a":NaN}')

    def test_boolean_reference_size_rejected(self):
        with self.assertRaises(ValueError):V.reference(dict(path=NAME,bytes=True,sha256='0'*64))

    def test_flat_names_bounded_and_distinct(self):
        one=V.local_name(V.D+'/'+'a'*90+'.json');two=V.local_name(V.A+'/'+'a'*90+'.json')
        self.assertLessEqual(len(one),77);self.assertNotEqual(one,two)

    def test_corrupt_header_checksum_rejected(self):
        raw=bytearray(raw_tar());raw[0]^=1
        with self.assertRaises((ValueError,tarfile.HeaderError)):self.parse_archive(bytes(raw))

    def test_noncanonical_unused_header_bytes_rejected(self):
        raw=bytearray(header());raw[500]=1;raw[148:156]=b' '*8
        checksum=sum(raw);raw[148:156]=('%06o\0 '%checksum).encode()
        with self.assertRaises(ValueError):self.parse_archive(bytes(raw)+b'x'+b'\0'*511+b'\0'*1024)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LocalArchiveCPUChecks))
    raise SystemExit(not result.wasSuccessful())
