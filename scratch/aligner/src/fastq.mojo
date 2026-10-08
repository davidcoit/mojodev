"""Streaming FASTQ reader that packs reads straight into 4-bit slots.

The file is consumed in large blocks (CHUNK_BYTES); records never straddle a refill
because the unread tail is moved to the front of the working buffer first.  The file
is reopened and seeked for each refill, so the reader holds no file handle.
"""
from nt4 import PU8, nt4_from_ascii, nt4_set
from kernels import MAXL

comptime CHUNK_BYTES = 64 * 1024 * 1024
comptime MIN_AVAIL = 16384  # bytes that must be buffered before parsing a record


struct FastqReader(Movable):
    var path: String
    var file_off: Int
    var tmp: List[UInt8]
    var buf: List[UInt8]
    var pos: Int
    var end: Int
    var eof: Bool

    def __init__(out self, path: String) raises:
        self.path = path
        self.file_off = 0
        self.tmp = List[UInt8](length=CHUNK_BYTES, fill=0)
        self.buf = List[UInt8](length=CHUNK_BYTES + MIN_AVAIL * 2, fill=0)
        self.pos = 0
        self.end = 0
        self.eof = False
        self.refill()

    def refill(mut self) raises:
        """Move the unread tail to the front and append the next block of the file."""
        var keep = self.end - self.pos
        var bp = self.buf.unsafe_ptr()
        for i in range(keep):
            bp[i] = bp[self.pos + i]
        var f = open(self.path, "r")
        _ = f.seek(self.file_off, 0)
        var got = Int(f.read(Span(self.tmp)))
        f.close()
        if got <= 0:
            self.eof = True
            got = 0
        var tp = self.tmp.unsafe_ptr()
        for i in range(got):
            bp[keep + i] = tp[i]
        self.file_off += got
        self.pos = 0
        self.end = keep + got

    def next_read(mut self, slot: PU8, mut names: List[UInt8]) raises -> Int:
        """Parse one record into `slot` (4-bit packed); append its name to `names`.

        Returns the read length, or -1 at end of file.  Reads longer than MAXL are
        consumed but only the first MAXL bases are stored (the caller marks them).
        """
        if self.end - self.pos < MIN_AVAIL and not self.eof:
            self.refill()
        if self.pos >= self.end:
            return -1
        var p = self.buf.unsafe_ptr()
        var i = self.pos + 1  # skip '@'
        var name_start = i
        while i < self.end and p[i] != 10 and p[i] != 32 and p[i] != 9:
            i += 1
        var name_end = i
        # drop a trailing /1 or /2
        if name_end - name_start > 2 and p[name_end - 2] == 47 and (p[name_end - 1] == 49 or p[name_end - 1] == 50):
            name_end -= 2
        for k in range(name_start, name_end):
            names.append(p[k])
        while i < self.end and p[i] != 10:
            i += 1
        i += 1
        var length = 0
        while i < self.end and p[i] != 10:
            if p[i] != 13:
                if length < MAXL:
                    nt4_set(slot, length, nt4_from_ascii(p[i]))
                length += 1
            i += 1
        i += 1
        # '+' line, then qualities (skipped)
        while i < self.end and p[i] != 10:
            i += 1
        i += 1
        while i < self.end and p[i] != 10:
            i += 1
        i += 1
        self.pos = i
        return length
