"""NRF24L01 driver for MicroPython
Source: micropython/micropython-lib (official driver)
"""

from micropython import const
import utime

CONFIG = const(0x00)
EN_RXADDR = const(0x02)
SETUP_AW = const(0x03)
SETUP_RETR = const(0x04)
RF_CH = const(0x05)
RF_SETUP = const(0x06)
STATUS = const(0x07)
RX_ADDR_P0 = const(0x0A)
TX_ADDR = const(0x10)
RX_PW_P0 = const(0x11)
FIFO_STATUS = const(0x17)
DYNPD = const(0x1C)

EN_CRC = const(0x08)
CRCO = const(0x04)
PWR_UP = const(0x02)
PRIM_RX = const(0x01)

POWER_0 = const(0x00)
POWER_1 = const(0x02)
POWER_2 = const(0x04)
POWER_3 = const(0x06)
SPEED_1M = const(0x00)
SPEED_2M = const(0x08)
SPEED_250K = const(0x20)

RX_DR = const(0x40)
TX_DS = const(0x20)
MAX_RT = const(0x10)

RX_EMPTY = const(0x01)

R_RX_PL_WID = const(0x60)
R_RX_PAYLOAD = const(0x61)
W_TX_PAYLOAD = const(0xA0)
FLUSH_TX = const(0xE1)
FLUSH_RX = const(0xE2)
NOP = const(0xFF)


class NRF24L01:
    def __init__(self, spi, cs, ce, channel=46, payload_size=16):
        assert payload_size <= 32
        self.buf = bytearray(1)
        self.spi = spi
        self.cs = cs
        self.ce = ce
        self.init_spi(4000000)
        ce.init(ce.OUT, value=0)
        cs.init(cs.OUT, value=1)
        self.payload_size = payload_size
        self.pipe0_read_addr = None
        utime.sleep_ms(5)

        self.reg_write(SETUP_AW, 0b11)
        if self.reg_read(SETUP_AW) != 0b11:
            raise OSError("nRF24L01+ Hardware not responding")

        self.reg_write(DYNPD, 0)
        self.reg_write(SETUP_RETR, (6 << 4) | 8)
        self.set_power_speed(POWER_3, SPEED_250K)
        self.set_crc(2)
        self.reg_write(STATUS, RX_DR | TX_DS | MAX_RT)
        self.set_channel(channel)
        self.flush_rx()
        self.flush_tx()

    def init_spi(self, baudrate):
        try:
            master = self.spi.MASTER
        except AttributeError:
            self.spi.init(baudrate=baudrate, polarity=0, phase=0)
        else:
            self.spi.init(master, baudrate=baudrate, polarity=0, phase=0)

    def reg_read(self, reg):
        self.cs(0)
        self.spi.readinto(self.buf, reg)
        self.spi.readinto(self.buf)
        self.cs(1)
        return self.buf[0]

    def reg_write_bytes(self, reg, buf):
        self.cs(0)
        self.spi.readinto(self.buf, 0x20 | reg)
        self.spi.write(buf)
        self.cs(1)
        return self.buf[0]

    def reg_write(self, reg, value):
        self.cs(0)
        self.spi.readinto(self.buf, 0x20 | reg)
        ret = self.buf[0]
        self.spi.readinto(self.buf, value)
        self.cs(1)
        return ret

    def read_status(self):
        self.cs(0)
        self.spi.readinto(self.buf, NOP)
        self.cs(1)
        return self.buf[0]

    def flush_rx(self):
        self.cs(0)
        self.spi.readinto(self.buf, FLUSH_RX)
        self.cs(1)

    def flush_tx(self):
        self.cs(0)
        self.spi.readinto(self.buf, FLUSH_TX)
        self.cs(1)

    def set_power_speed(self, power, speed):
        setup = self.reg_read(RF_SETUP) & 0b11010001
        self.reg_write(RF_SETUP, setup | power | speed)

    def set_crc(self, length):
        config = self.reg_read(CONFIG) & ~(CRCO | EN_CRC)
        if length == 0:
            pass
        elif length == 1:
            config |= EN_CRC
        else:
            config |= EN_CRC | CRCO
        self.reg_write(CONFIG, config)

    def set_channel(self, channel):
        self.reg_write(RF_CH, min(channel, 125))

    def open_tx_pipe(self, address):
        assert len(address) == 5
        self.reg_write_bytes(RX_ADDR_P0, address)
        self.reg_write_bytes(TX_ADDR, address)
        self.reg_write(RX_PW_P0, self.payload_size)

    def open_rx_pipe(self, pipe_id, address):
        assert len(address) == 5
        assert 0 <= pipe_id <= 5
        if pipe_id == 0:
            self.pipe0_read_addr = address
        if pipe_id < 2:
            self.reg_write_bytes(RX_ADDR_P0 + pipe_id, address)
        else:
            self.reg_write(RX_ADDR_P0 + pipe_id, address[0])
        self.reg_write(RX_PW_P0 + pipe_id, self.payload_size)
        self.reg_write(EN_RXADDR, self.reg_read(EN_RXADDR) | (1 << pipe_id))

    def start_listening(self):
        self.reg_write(CONFIG, self.reg_read(CONFIG) | PWR_UP | PRIM_RX)
        self.reg_write(STATUS, RX_DR | TX_DS | MAX_RT)
        if self.pipe0_read_addr is not None:
            self.reg_write_bytes(RX_ADDR_P0, self.pipe0_read_addr)
        self.flush_rx()
        self.flush_tx()
        self.ce(1)
        utime.sleep_us(130)

    def stop_listening(self):
        self.ce(0)
        self.flush_tx()
        self.flush_rx()

    def any(self):
        return not bool(self.reg_read(FIFO_STATUS) & RX_EMPTY)

    def recv(self):
        self.cs(0)
        self.spi.readinto(self.buf, R_RX_PAYLOAD)
        buf = self.spi.read(self.payload_size)
        self.cs(1)
        self.reg_write(STATUS, RX_DR)
        return buf

    def send(self, buf, timeout=500):
        self.send_start(buf)
        start = utime.ticks_ms()
        result = None
        while result is None and utime.ticks_diff(utime.ticks_ms(), start) < timeout:
            result = self.send_done()
        if result is None:
            self.flush_tx()
            self.reg_write(CONFIG, self.reg_read(CONFIG) & ~PWR_UP)
            raise OSError("timed out")
        if result == 2:
            raise OSError("send failed")

    def send_start(self, buf):
        self.reg_write(CONFIG, (self.reg_read(CONFIG) | PWR_UP) & ~PRIM_RX)
        utime.sleep_us(1500)
        self.cs(0)
        self.spi.readinto(self.buf, W_TX_PAYLOAD)
        self.spi.write(buf)
        if len(buf) < self.payload_size:
            self.spi.write(b"\x00" * (self.payload_size - len(buf)))
        self.cs(1)
        self.ce(1)
        utime.sleep_us(15)
        self.ce(0)

    def send_done(self):
        status = self.read_status()
        if not (status & (TX_DS | MAX_RT)):
            return None
        status = self.reg_write(STATUS, RX_DR | TX_DS | MAX_RT)
        self.reg_write(CONFIG, self.reg_read(CONFIG) & ~PWR_UP)
        return 1 if status & TX_DS else 2
