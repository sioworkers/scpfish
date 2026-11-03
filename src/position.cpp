/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "position.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>

#include "bitboard.h"
#include "history.h"
#include "misc.h"
#include "movegen.h"
#include "syzygy/tbprobe.h"
#include "uci.h"

using std::string;

namespace Stockfish {

using namespace Attacks;

namespace Zobrist {

Key psq[PIECE_NB][SQUARE_NB];
Key enpassant[FILE_NB];
Key castling[CASTLING_RIGHT_NB];
Key side, noPawns;
Key scp[SQUARE_NB];
Key scpOutcome;
Key zombie[SQUARE_NB];
Key phasePending;

}

namespace {

constexpr std::string_view PieceToChar(" PNBRQK  pnbrqk");

static constexpr Piece Pieces[] = {W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
                                   B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING};
}  // namespace


// Returns an ASCII representation of the position
std::ostream& operator<<(std::ostream& os, const Position& pos) {

    os << "\n +---+---+---+---+---+---+---+---+\n";

    for (Rank r = RANK_8;; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
            os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];

        os << " | " << (1 + r) << "\n +---+---+---+---+---+---+---+---+\n";

        if (r == RANK_1)
            break;
    }

    os << "   a   b   c   d   e   f   g   h\n"
       << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase << std::setfill('0')
       << std::setw(16) << pos.key() << std::setfill(' ') << std::dec << "\nCheckers: ";

    for (Bitboard b = pos.checkers(); b;)
        os << UCIEngine::square(pop_lsb(b)) << " ";

    if (pos.scp_bb())
        os << (pos.scp_type() == SCP_049 ? "\nSCP-049: " : "\nSCP-096: ")
           << UCIEngine::square(pos.scp_square());

    if (pos.scp_type() == SCP_049)
    {
        os << "\nZombies:";
        for (Bitboard z = pos.zombies(); z;)
        {
            const Square s = pop_lsb(z);
            os << " " << UCIEngine::square(s);
        }
        os << "\nZombie moves:";
        for (Bitboard z = pos.zombies(); z;)
        {
            const Square s = pop_lsb(z);
            os << " " << UCIEngine::square(s) << ":";
            for (Bitboard t = pos.scp049_zombie_moves(s); t;)
                os << UCIEngine::square(pop_lsb(t)) << (t ? "," : "");
        }
        if (pos.phase_pending())
            os << "\nPhase pending: yes";
    }

    if (Square t = pos.scp_victim(); t != SQ_NONE && !pos.scp_outcome())
        os << "\nSCP target: " << UCIEngine::square(t);

    if (pos.scp_outcome())
        os << "\nSCP outcome: " << (pos.scp_outcome() > 0 ? "side to move wins" : "side to move loses");

    if (Tablebases::MaxCardinality >= popcount(pos.pieces()) && !pos.can_castle(ANY_CASTLING)
        && !pos.scp_bb())
    {
        StateInfo st;

        Position p;
        p.set(pos.fen(), pos.is_chess960(), &st);
        Tablebases::ProbeState s1, s2;
        Tablebases::WDLScore   wdl = Tablebases::probe_wdl(p, &s1);
        int                    dtz = Tablebases::probe_dtz(p, &s2);
        os << "\nTablebases WDL: " << std::setw(4) << wdl << " (" << s1 << ")"
           << "\nTablebases DTZ: " << std::setw(4) << dtz << " (" << s2 << ")";
    }

    return os;
}


// Implements Marcel van Kervinck's cuckoo algorithm to detect repetition of positions
// for 3-fold repetition draws. The algorithm uses two hash tables with Zobrist hashes
// to allow fast detection of recurring positions. For details see:
// http://web.archive.org/web/20201107002606/https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf

// First and second hash functions for indexing the cuckoo tables
inline int H1(Key h) { return h & 0x1fff; }
inline int H2(Key h) { return (h >> 16) & 0x1fff; }

// Cuckoo tables with Zobrist hashes of valid reversible moves, and the moves themselves
static std::array<Key, 8192>  cuckoo;
static std::array<Move, 8192> cuckooMove;

// Initializes at startup the various arrays used to compute hash keys
void Position::init() {

    PRNG rng(1070372);

    for (Piece pc : Pieces)
        for (Square s = SQ_A1; s <= SQ_H8; ++s)
            Zobrist::psq[pc][s] = rng.rand<Key>();
    // pawns on these squares will promote
    std::fill_n(Zobrist::psq[W_PAWN] + SQ_A8, 8, 0);
    std::fill_n(Zobrist::psq[B_PAWN], 8, 0);

    for (File f = FILE_A; f <= FILE_H; ++f)
        Zobrist::enpassant[f] = rng.rand<Key>();

    for (int cr = NO_CASTLING; cr <= ANY_CASTLING; ++cr)
        Zobrist::castling[cr] = rng.rand<Key>();

    Zobrist::side    = rng.rand<Key>();
    Zobrist::noPawns = rng.rand<Key>();

    for (Square s = SQ_A1; s <= SQ_H8; ++s)
        Zobrist::scp[s] = rng.rand<Key>();
    Zobrist::scpOutcome = rng.rand<Key>();

    for (Square s = SQ_A1; s <= SQ_H8; ++s)
        Zobrist::zombie[s] = rng.rand<Key>();
    Zobrist::phasePending = rng.rand<Key>();

    // Prepare the cuckoo tables
    cuckoo.fill(0);
    cuckooMove.fill(Move::none());
    [[maybe_unused]] int count = 0;
    for (Piece pc : Pieces)
        for (Square s1 = SQ_A1; s1 <= SQ_H8; ++s1)
            for (Square s2 = Square(s1 + 1); s2 <= SQ_H8; ++s2)
                if ((type_of(pc) != PAWN) && (attacks_bb(type_of(pc), s1, 0) & s2))
                {
                    Move move = Move(s1, s2);
                    Key  key  = Zobrist::psq[pc][s1] ^ Zobrist::psq[pc][s2] ^ Zobrist::side;
                    int  i    = H1(key);
                    while (true)
                    {
                        std::swap(cuckoo[i], key);
                        std::swap(cuckooMove[i], move);
                        if (move == Move::none())  // Arrived at empty slot?
                            break;
                        i = (i == H1(key)) ? H2(key) : H1(key);  // Push victim to alternative slot
                    }
                    count++;
                }
    assert(count == 3668);
}


// Initializes the position object with the given FEN string.
// The FEN string is strictly validated; if it is invalid or inconsistent,
// a PositionSetError describing the problem is returned, otherwise std::nullopt.
std::optional<PositionSetError>
Position::set(const string& fenStr, bool isChess960, StateInfo* si) {
    /*
   A FEN string defines a particular position using only the ASCII character set.

   A FEN string contains six fields separated by a space. The fields are:

   1) Piece placement (from white's perspective). Each rank is described, starting
      with rank 8 and ending with rank 1. Within each rank, the contents of each
      square are described from file A through file H. Following the Standard
      Algebraic Notation (SAN), each piece is identified by a single letter taken
      from the standard English names. White pieces are designated using upper-case
      letters ("PNBRQK") whilst Black uses lowercase ("pnbrqk"). Blank squares are
      noted using digits 1 through 8 (the number of blank squares), and "/"
      separates ranks.

   2) Active color. "w" means white moves next, "b" means black.

   3) Castling availability. If neither side can castle, this is "-". Otherwise,
      this has one or more letters: "K" (White can castle kingside), "Q" (White
      can castle queenside), "k" (Black can castle kingside), and/or "q" (Black
      can castle queenside).

   4) En passant target square (in algebraic notation). If there's no en passant
      target square, this is "-". If a pawn has just made a 2-square move, this
      is the position "behind" the pawn. Following X-FEN standard, this is recorded
      only if there is a pawn in position to make an en passant capture, and if
      there really is a pawn that might have advanced two squares.

   5) Halfmove clock. This is the number of halfmoves since the last pawn advance
      or capture. This is used to determine if a draw can be claimed under the
      fifty-move rule.

   6) Fullmove number. The number of the full move. It starts at 1, and is
      incremented after Black's move.

   7) Optional SCP fields. Without them there is no SCP on the board and the
      position is played as regular chess.
      - "scp:<square>", e.g. "scp:a4": SCP-096 occupies that square.
      - "scp049:<square>", e.g. "scp049:b4": SCP-049 occupies that square, and
        "zombies:<square>,<square>,..." lists the pieces it has turned into zombies.
*/

    unsigned char      token;
    std::istringstream ss(fenStr);

    std::memset(reinterpret_cast<char*>(this), 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    // 7. SCP fields, parsed first as en passant validation depends on them
    Bitboard zombieSquares = 0;
    {
        std::istringstream fields(fenStr);
        string             field;
        Square             scpSq = SQ_NONE;

        auto parse_square = [](const string& str, usize i) {
            return i + 1 < str.size() && str[i] >= 'a' && str[i] <= 'h' && str[i + 1] >= '1'
                  && str[i + 1] <= '8'
                   ? make_square(File(str[i] - 'a'), Rank(str[i + 1] - '1'))
                   : SQ_NONE;
        };

        while (fields >> field)
        {
            const bool is096 = field.rfind("scp:", 0) == 0;
            const bool is049 = field.rfind("scp049:", 0) == 0;

            if (is096 || is049)
            {
                const usize i = is096 ? 4 : 7;
                scpSq         = parse_square(field, i);
                if (scpSq == SQ_NONE || field.size() != i + 2)
                    return PositionSetError("Invalid FEN. Invalid SCP square: " + field);
                scpType = is096 ? SCP_096 : SCP_049;
            }
            else if (field.rfind("zombies:", 0) == 0)
                for (usize i = 8; i < field.size(); i += 3)
                {
                    const Square z = parse_square(field, i);
                    if (z == SQ_NONE || (i + 2 < field.size() && field[i + 2] != ','))
                        return PositionSetError("Invalid FEN. Invalid zombie list: " + field);
                    zombieSquares |= z;
                }
        }

        if (zombieSquares && scpType != SCP_049)
            return PositionSetError("Invalid FEN. Zombies need SCP-049 on the board.");

        set_scp(scpSq);
    }

    ss >> std::noskipws;

    int numPieces = 0;
    int file      = FILE_A;
    int rank      = RANK_8;

    // 1. Piece placement
    for (;;)
    {
        if (!(ss >> token))
            return PositionSetError("Invalid FEN. Unexpected end of stream.");

        if (isspace(token))
            break;

        if (isdigit(token))
        {
            const int diff = (token - '0');
            if (diff < 1 || diff > 8)
                return PositionSetError("Invalid FEN. Invalid number of squares to skip.");

            file += diff;
            if (file > FILE_NB)
                return PositionSetError("Invalid FEN. Invalid file reached.");
        }
        else if (token == '/')
        {
            if (file != FILE_NB)
                return PositionSetError(
                  "Invalid FEN. Trying to end rank when not at the end of it.");

            --rank;
            file = FILE_A;

            if (rank < RANK_1)
                return PositionSetError("Invalid FEN. Invalid rank reached.");
        }
        else
        {
            if (file >= FILE_NB)
                return PositionSetError("Invalid FEN. Invalid file reached.");

            const usize idx = PieceToChar.find(token);
            if (idx == string::npos)
                return PositionSetError(std::string("Invalid FEN. Invalid piece: ")
                                        + std::string(1, token));

            if (++numPieces > 32)
                return PositionSetError("Invalid FEN. More than 32 pieces on the board.");

            const Square sq = make_square(File(file), Rank(rank));
            put_piece(Piece(idx), sq);

            ++file;
        }
    }

    if (rank != RANK_1 || file != FILE_NB)
        return PositionSetError("Invalid FEN. Board state encoding ended but cursor not at end.");

    if (pieces(PAWN) & (Rank1BB | Rank8BB))
        return PositionSetError("Unsupported position. Pawns on the first or eighth rank.");

    if (count<KING>(WHITE) != 1 || count<KING>(BLACK) != 1)
        return PositionSetError("Unsupported position. Incorrect number of kings.");

    if (pieces() & st->scpBB)
        return PositionSetError("Invalid FEN. The SCP square is occupied.");

    if (zombieSquares & ~pieces())
        return PositionSetError("Invalid FEN. A zombie square is empty.");

    st->zombieBB = zombieSquares;

    for (Color c : {WHITE, BLACK})
    {
        if (count<PAWN>(c) > 8)
            return PositionSetError(std::string("Unsupported position. ")
                                    + (c == WHITE ? "WHITE" : "BLACK") + " has more than 8 pawns.");

        int additional = std::max(count<KNIGHT>(c) - 2, 0) + std::max(count<BISHOP>(c) - 2, 0)
                       + std::max(count<ROOK>(c) - 2, 0) + std::max(count<QUEEN>(c) - 1, 0);
        if (additional > 8 - count<PAWN>(c))
            return PositionSetError(std::string("Unsupported position. Too many pieces for ")
                                    + (c == WHITE ? "WHITE." : "BLACK."));
    }

    // 2. Active color
    if (!(ss >> token))
        return PositionSetError("Invalid FEN. Unexpected end of stream.");
    if (token != 'w' && token != 'b')
        return PositionSetError(std::string("Invalid FEN. Invalid side to move: ")
                                + std::string(1, token));
    sideToMove = (token == 'w' ? WHITE : BLACK);
    if (!(ss >> token) || !isspace(token) || ss.eof())
        return PositionSetError("Invalid FEN. Expected whitespace after side to move.");

    // 3. Castling availability. Compatible with 3 standards: Normal FEN standard,
    // Shredder-FEN that uses the letters of the columns on which the rooks began
    // the game instead of KQkq and also X-FEN standard that, in case of Chess960,
    // if an inner rook is associated with the castling right, the castling tag is
    // replaced by the file letter of the involved rook, as for the Shredder-FEN.
    //
    // NOTE: Due to the prevalence of incorrect (or missing) castling rights the
    // validation is less strict. However, incorrect castling rights are still sanitized.
    int num_castling_rights = 0;
    for (;;)
    {
        if (!(ss >> token))
            break;

        if (isspace(token))
            break;

        if (num_castling_rights == 0 && token == '-')
        {
            ss >> std::ws;
            break;
        }

        if (++num_castling_rights > 4)
            return PositionSetError("Invalid FEN. Maximum of 4 castling rights can be specified.");

        Square rsq  = SQ_NONE;
        Square ksq  = SQ_NONE;
        Color  c    = islower(token) ? BLACK : WHITE;
        Piece  rook = make_piece(c, ROOK);
        Piece  king = make_piece(c, KING);

        token = char(toupper(token));

        if (token == 'K' || token == 'Q')
        {
            const int dir = token == 'K' ? -1 : 1;
            Square    sq  = relative_square(c, token == 'K' ? SQ_H1 : SQ_A1);
            // Look for a rook and a king for the castling. King must come later.
            // Only the first rook is noted.
            // If the castling rights are available the king must always be between files 2 and 7 inclusive
            // so there is no need to check the last square.
            for (int i = 0; i < 7; ++i, sq = Square(sq + dir))
            {
                const Piece pc = piece_on(sq);
                if (pc == king)
                {
                    ksq = sq;
                    break;
                }
                else if (pc == rook && rsq == SQ_NONE)
                {
                    rsq = sq;
                }
            }
        }
        else if (token >= 'A' && token <= 'H')
        {
            const Square rsqCandidate = make_square(File(token - 'A'), relative_rank(c, RANK_1));
            if (piece_on(rsqCandidate) == rook)
                rsq = rsqCandidate;

            // If the castling rights are available the king must always be between files 2 and 7 inclusive.
            Square sq = relative_square(c, SQ_B1);
            for (int i = 0; i < 6; ++i, ++sq)
            {
                if (piece_on(sq) == king)
                    ksq = sq;
            }
        }
        else
        {
            return PositionSetError(std::string("Invalid FEN. Expected castling rights. Got: ")
                                    + std::string(1, token));
        }

        // SCP variants castle only with the king on e1 and the rook in its corner
        if (scpType != SCP_NONE_TYPE && !isChess960 && (token == 'K' || token == 'Q')
            && (ksq != relative_square(c, SQ_E1)
                || rsq != relative_square(c, token == 'K' ? SQ_H1 : SQ_A1)))
            rsq = SQ_NONE;

        // Only apply castling rights if they can be valid.
        if (ksq != SQ_NONE && rsq != SQ_NONE)
            set_castling_right(c, rsq);
    }

    // 4. En passant square.
    // Ignore if square is invalid or not on side to move relative rank 6.
    bool          enpassant = false, legalEP = false;
    unsigned char col = '-', row;
    ss >> col;
    if (col != '-')
    {
        if (!(ss >> row))
            return PositionSetError("Invalid FEN. Unexpected end of stream.");

        if ((col >= 'a' && col <= 'h') && (row == (sideToMove == WHITE ? '6' : '3')))
        {
            st->epSquare = make_square(File(col - 'a'), Rank(row - '1'));

            Bitboard pawns = attacks_bb(PAWN, st->epSquare, ~sideToMove) & pieces(sideToMove, PAWN);
            Bitboard target = (pieces(~sideToMove, PAWN) & (st->epSquare + pawn_push(~sideToMove)));
            Bitboard occ    = blockers() ^ target ^ st->epSquare;

            // En passant square will be considered only if
            // a) side to move have a pawn threatening epSquare
            // b) there is an enemy pawn in front of epSquare
            // c) there is no piece on epSquare or behind epSquare
            enpassant = pawns && target
                     && !(blockers() & (st->epSquare | (st->epSquare + pawn_push(sideToMove))))
                     && !(st->scpKingForbidden & square<KING>(sideToMove));

            // SCP-049 has no pins, and zombie pawns take no part in en passant
            if (scpType == SCP_049)
            {
                // A zombie may have moved behind the en passant square
                pawns &= ~st->zombieBB;
                enpassant = pawns && target && !(target & st->zombieBB)
                         && !(blockers() & st->epSquare);
                legalEP   = true;
                pawns     = 0;
            }

            // If no pawn can execute the en passant capture without leaving the king in check, don't record the epSquare
            while (pawns)
                legalEP |= !(attackers_to(square<KING>(sideToMove), occ ^ pop_lsb(pawns))
                             & pieces(~sideToMove) & ~target);
        }
        else
            return PositionSetError("Invalid FEN. Invalid en-passant square.");
    }

    if (!enpassant || !legalEP)
        st->epSquare = SQ_NONE;

    // 5-6. Halfmove clock and fullmove number
    ss >> std::skipws >> st->rule50 >> gamePly;

    // Normally values larger than 99 would be pointless but we do support ignoring 50 move rule for TB purposes.
    // Limit at 2**15 as it's used multiplicatively with position evaluation during search.
    if (st->rule50 < 0 || st->rule50 > 32767)
        return PositionSetError("Unsupported position. Rule50 counter out of range.");

    if (gamePly < 0 || gamePly > 100000)
        return PositionSetError("Unsupported position. Game ply out of range.");

    // Convert from fullmove starting from 1 to gamePly starting from 0,
    // handle also common incorrect FEN with fullmove = 0.
    gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);

    chess960 = isChess960;
    set_state();

    // Under SCP-049 a capturable king is legal: the side to move has won
    if (scpType != SCP_049 && (attackers_to(square<KING>(~sideToMove)) & pieces(sideToMove)))
        return PositionSetError("Unsupported position. King can be captured.");

    assert(pos_is_ok());

    return std::nullopt;
}


// Helper function used to set castling
// rights given the corresponding color and the rook starting square.
void Position::set_castling_right(Color c, Square rfrom) {

    Square         kfrom = square<KING>(c);
    CastlingRights cr    = c & (kfrom < rfrom ? KING_SIDE : QUEEN_SIDE);

    st->castlingRights |= cr;
    castlingRightsMask[kfrom] |= cr;
    castlingRightsMask[rfrom] |= cr;
    castlingRookSquare[cr] = rfrom;

    Square kto = relative_square(c, cr & KING_SIDE ? SQ_G1 : SQ_C1);
    Square rto = relative_square(c, cr & KING_SIDE ? SQ_F1 : SQ_D1);

    castlingPath[cr] = (between_bb(rfrom, rto) | between_bb(kfrom, kto)) & ~(kfrom | rfrom);
}


// Sets pinners and where stm can give check. Squares threatened by ~stm are
// computed lazily by compute_threats().
void Position::set_extra_bitboards() const {

    update_slider_blockers(WHITE);
    update_slider_blockers(BLACK);

    Square ksq                              = square<KING>(~sideToMove);
    const auto [bishopAttacks, rookAttacks] = both_attacks_bb(ksq, blockers());

    st->checkSquares[PAWN]   = attacks_bb(PAWN, ksq, ~sideToMove);
    st->checkSquares[KNIGHT] = attacks_bb(KNIGHT, ksq);
    st->checkSquares[BISHOP] = bishopAttacks;
    st->checkSquares[ROOK]   = rookAttacks;
    st->checkSquares[QUEEN]  = st->checkSquares[BISHOP] | st->checkSquares[ROOK];
    st->checkSquares[KING]   = 0;

    // A zombie king cannot be checked
    if (st->zombieBB & ksq)
        for (PieceType pt = PAWN; pt <= KING; ++pt)
            st->checkSquares[pt] = 0;

    st->threats[ALL_PIECES] = 0;
}


void Position::compute_threats() const {

    Bitboard occ            = blockers() ^ square<KING>(sideToMove);
    st->threats[ALL_PIECES] = st->threats[PAWN]    = xattacks_by(PAWN, ~sideToMove, occ);
    st->threats[ALL_PIECES] |= st->threats[KNIGHT] = xattacks_by(KNIGHT, ~sideToMove, occ);
    st->threats[ALL_PIECES] |= st->threats[BISHOP] = xattacks_by(BISHOP, ~sideToMove, occ);
    st->threats[ALL_PIECES] |= st->threats[ROOK]   = xattacks_by(ROOK, ~sideToMove, occ);
    st->threats[ALL_PIECES] |= st->threats[QUEEN]  = xattacks_by(QUEEN, ~sideToMove, occ);
    st->threats[ALL_PIECES] |= st->threats[KING]   = xattacks_by(KING, ~sideToMove, occ);
}


// Computes the hash keys of the position, and other
// data that once computed is updated incrementally as moves are made.
// The function is only used when a new position is set up
void Position::set_state() const {

    st->key               = 0;
    st->scpOutcome        = SCP_NONE;
    st->scpCaptured       = NO_PIECE;
    st->minorPieceKey     = 0;
    st->nonPawnKey[WHITE] = st->nonPawnKey[BLACK] = 0;
    st->pawnKey                                   = Zobrist::noPawns;
    st->nonPawnMaterial[WHITE] = st->nonPawnMaterial[BLACK] = VALUE_ZERO;
    st->checkersBB = attackers_to(square<KING>(sideToMove)) & pieces(~sideToMove);

    // A king a knight's move from SCP-096 is in check
    if (st->scpKingForbidden & square<KING>(sideToMove))
        st->checkersBB |= st->scpBB;

    if (st->scpBB)
        st->key ^= Zobrist::scp[lsb(st->scpBB)];

    for (Bitboard z = st->zombieBB; z;)
        st->key ^= Zobrist::zombie[pop_lsb(z)];

    if (st->phasePending)
        st->key ^= Zobrist::phasePending;

    set_extra_bitboards();

    if (scpType == SCP_049)
        scp049_check_info();

    for (Bitboard b = pieces(); b;)
    {
        Square s  = pop_lsb(b);
        Piece  pc = piece_on(s);
        st->key ^= Zobrist::psq[pc][s];

        if (type_of(pc) == PAWN)
            st->pawnKey ^= Zobrist::psq[pc][s];

        else
        {
            st->nonPawnKey[color_of(pc)] ^= Zobrist::psq[pc][s];

            if (type_of(pc) != KING)
            {
                st->nonPawnMaterial[color_of(pc)] += PieceValue[pc];

                if (type_of(pc) <= BISHOP)
                    st->minorPieceKey ^= Zobrist::psq[pc][s];
            }
        }
    }

    if (st->epSquare != SQ_NONE)
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];

    if (sideToMove == BLACK)
        st->key ^= Zobrist::side;

    st->key ^= Zobrist::castling[st->castlingRights];
    st->materialKey = compute_material_key();
}

Key Position::compute_material_key() const {
    Key k = 0;
    for (Piece pc : Pieces)
        for (int cnt = 0; cnt < pieceCount[pc]; ++cnt)
            k ^= Zobrist::psq[pc][8 + cnt];
    return k;
}


// Overload to initialize the position object with the given endgame code string
// like "KBPKN". It's mainly a helper to get the material key out of an endgame code.
std::optional<PositionSetError> Position::set(const string& code, Color c, StateInfo* si) {

    assert(code[0] == 'K');

    string sides[] = {code.substr(code.find('K', 1)),                                // Weak
                      code.substr(0, std::min(code.find('v'), code.find('K', 1)))};  // Strong

    assert(sides[0].length() > 0 && sides[0].length() < 8);
    assert(sides[1].length() > 0 && sides[1].length() < 8);

    std::transform(sides[c].begin(), sides[c].end(), sides[c].begin(), tolower);

    string fenStr = "8/" + sides[0] + char(8 - sides[0].length() + '0') + "/8/8/8/8/" + sides[1]
                  + char(8 - sides[1].length() + '0') + "/8 w - - 0 10";

    return set(fenStr, false, si);
}


// Returns a FEN representation of the position. In case of
// Chess960 the Shredder-FEN notation is used. This is mainly a debugging function.
string Position::fen() const {

    int                emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_8;; --r)
    {
        for (File f = FILE_A; f <= FILE_H; ++f)
        {
            for (emptyCnt = 0; f <= FILE_H && empty(make_square(f, r)); ++f)
                ++emptyCnt;

            if (emptyCnt)
                ss << emptyCnt;

            if (f <= FILE_H)
                ss << PieceToChar[piece_on(make_square(f, r))];
        }

        if (r == RANK_1)
            break;
        ss << '/';
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    if (can_castle(WHITE_OO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OO))) : 'K');

    if (can_castle(WHITE_OOO))
        ss << (chess960 ? char('A' + file_of(castling_rook_square(WHITE_OOO))) : 'Q');

    if (can_castle(BLACK_OO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OO))) : 'k');

    if (can_castle(BLACK_OOO))
        ss << (chess960 ? char('a' + file_of(castling_rook_square(BLACK_OOO))) : 'q');

    if (!can_castle(ANY_CASTLING))
        ss << '-';

    ss << (ep_square() == SQ_NONE ? " - " : " " + UCIEngine::square(ep_square()) + " ")
       << st->rule50 << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    if (st->scpBB)
        ss << (scpType == SCP_049 ? " scp049:" : " scp:") << UCIEngine::square(scp_square());

    if (st->zombieBB)
    {
        ss << " zombies:";
        for (Bitboard z = st->zombieBB; z;)
            ss << UCIEngine::square(pop_lsb(z)) << (z ? "," : "");
    }

    return ss.str();
}

// Calculates st->blockersForKing[c] and st->pinners[~c],
// which store respectively the pieces preventing king of color c from being in check
// and the slider pieces of color ~c pinning pieces of color c to the king.
void Position::update_slider_blockers(Color c) const {

    Square ksq = square<KING>(c);

    st->blockersForKing[c] = 0;
    st->pinners[~c]        = 0;

    // Snipers are sliders that attack 's' when a piece and other snipers are removed
    // Zombies do not capture, and a zombie king cannot be captured
    if (st->zombieBB & ksq)
        return;

    Bitboard snipers = ((attacks_bb(ROOK, ksq) & pieces(QUEEN, ROOK))
                        | (attacks_bb(BISHOP, ksq) & pieces(QUEEN, BISHOP)))
                     & pieces(~c) & ~st->zombieBB;
    Bitboard occupancy = blockers() ^ snipers;  // the SCP blocks lines

    while (snipers)
    {
        Square   sniperSq = pop_lsb(snipers);
        Bitboard b        = between_bb(ksq, sniperSq) & occupancy;

        if (b && !more_than_one(b))
        {
            st->blockersForKing[c] |= b & pieces();
            if (b & pieces(c))
                st->pinners[~c] |= sniperSq;
        }
    }
}


// Computes a bitboard of all pieces which attack a given square.
// Slider attacks use the occupied bitboard to indicate occupancy.
Bitboard Position::attackers_to(Square s, Bitboard occupied) const {

    const auto [bishopAttacks, rookAttacks] = both_attacks_bb(s, occupied);

    return (rookAttacks & pieces(ROOK, QUEEN)) | (bishopAttacks & pieces(BISHOP, QUEEN))
         | (attacks_bb(PAWN, s, BLACK) & pieces(WHITE, PAWN))
         | (attacks_bb(PAWN, s, WHITE) & pieces(BLACK, PAWN))
         | (attacks_bb(KNIGHT, s) & pieces(KNIGHT)) | (attacks_bb(KING, s) & pieces(KING));
}

// Takes a random move and tests whether the move is legal. It is used to validate moves
// from TT that can be corrupted due to SMP concurrent access or hash position key aliasing.
bool Position::legal(Move m) const {

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = moved_piece(m);

    // Use a slower but simpler function for uncommon cases and for SCP-049
    if (m.type_of() != NORMAL || scpType == SCP_049)
        return MoveList<ALL>(*this).contains(m);

    // Is not a promotion, so the promotion piece must be empty
    assert(m.promotion_type() - KNIGHT == NO_PIECE_TYPE);

    // If the 'from' square is not occupied by a piece belonging to the side to
    // move, the move is obviously not legal.
    if (pc == NO_PIECE || color_of(pc) != us)
        return false;

    // The destination square cannot hold a friendly piece or an SCP
    if ((pieces(us) | st->scpBB) & to)
        return false;

    // Handle the special case of a pawn move
    if (type_of(pc) == PAWN)
    {
        // We have already handled promotion moves, so destination cannot be on the 8th/1st rank
        if ((Rank8BB | Rank1BB) & to)
            return false;

        // Check if it's a valid capture, single push, or double push
        const bool isCapture    = bool(attacks_bb(PAWN, from, us) & pieces(~us) & to);
        const bool isSinglePush = (from + pawn_push(us) == to) && empty(to);
        const bool isDoublePush = (from + 2 * pawn_push(us) == to)
                               && (relative_rank(us, from) == RANK_2) && empty(to)
                               && !(blockers() & (to - pawn_push(us)));

        if (!(isCapture || isSinglePush || isDoublePush))
            return false;
    }
    else if (!(attacks_bb(type_of(pc), from, blockers()) & to))
        return false;


    if (type_of(pc) != KING)
    {
        if (checkers())
        {
            // In double check, only a king move can evade
            if (more_than_one(checkers()))
                return false;

            // The move must block the check or capture the checker
            if (!(between_bb(square<KING>(us), lsb(checkers())) & to))
                return false;
        }
        // When not in check, a non-king move is legal if and only if it is not pinned
        // or it is moving along the ray towards or away from the king.
        return !(blockers_for_king(us) & from) || line_bb(from, to) & pieces(us, KING);
    }
    else
        // If the moving piece is a king, check whether the destination square is
        // attacked by the opponent or a knight's move away from SCP-096.
        return !(to & (threats_by(ALL_PIECES) | st->scpKingForbidden));
}


// Tests whether a legal move gives a check
bool Position::gives_check(Move m) const {

    assert(m.is_ok());
    assert(color_of(moved_piece(m)) == sideToMove);

    Square from = m.from_sq();
    Square to   = m.to_sq();

    // Is there a direct check?
    if (check_squares(type_of(piece_on(from))) & to)
        return true;

    // Is there a discovered check?
    if (blockers_for_king(~sideToMove) & from)
        return !(line_bb(from, to) & pieces(~sideToMove, KING)) || m.type_of() == CASTLING;

    switch (m.type_of())
    {
    case NORMAL :
        return false;

    case PROMOTION :
        return attacks_bb(m.promotion_type(), to, blockers() ^ from) & pieces(~sideToMove, KING);

    // En passant capture with check? We have already handled the case of direct
    // checks and ordinary discovered check, so the only case we need to handle
    // is the unusual case of a discovered check through the captured pawn.
    case EN_PASSANT : {
        Square   capsq                          = make_square(file_of(to), rank_of(from));
        Bitboard b                              = (blockers() ^ from ^ capsq) | to;
        const auto [bishopAttacks, rookAttacks] = both_attacks_bb(square<KING>(~sideToMove), b);

        return (rookAttacks & pieces(sideToMove, QUEEN, ROOK))
             | (bishopAttacks & pieces(sideToMove, QUEEN, BISHOP));
    }
    default :  //CASTLING
    {
        // Castling is encoded as 'king captures the rook'
        Square rto = relative_square(sideToMove, to > from ? SQ_F1 : SQ_D1);

        return check_squares(ROOK) & rto;
    }
    }
}


// Makes a move, and saves all information necessary
// to a StateInfo object. The move is assumed to be legal.
// If a pointer to the shared histories is passed, their entries for the
// new position will be prefetched.
void Position::do_move(
  Move m, StateInfo& newSt, bool givesCheck, Dirties& dirties, const SharedHistories* history) {

    assert(m.is_ok());
    assert(&newSt != st);

    Key k = st->key ^ Zobrist::side;

    // Copy some fields of the old state to our new StateInfo object except the
    // ones which are going to be recalculated from scratch anyway and then switch
    // our state pointer to point to the new (ready to be updated) state.
    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st             = &newSt;

    // Increment ply counters. In particular, rule50 will be reset to zero later on
    // in case of a capture or a pawn move.
    ++gamePly;
    ++st->rule50;
    ++st->pliesFromNull;

    auto& dpps = dirties.dirtyPawnPairs;
    auto& dts  = dirties.dirtyThreats;
    auto& dp   = dirties.dirtyPiece;

    dpps.before[WHITE] = pieces(WHITE, PAWN);
    dpps.before[BLACK] = pieces(BLACK, PAWN);

    Color  us       = sideToMove;
    Color  them     = ~us;
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = m.type_of() == EN_PASSANT ? make_piece(them, PAWN) : piece_on(to);

    dp.pc         = pc;
    dp.from       = from;
    dp.to         = to;
    dp.add_sq     = SQ_NONE;
    dp.scpCapture = false;

    assert(color_of(pc) == us);
    assert(captured == NO_PIECE || color_of(captured) == (m.type_of() != CASTLING ? them : us));
    assert(type_of(captured) != KING);

    if (m.type_of() == CASTLING)
    {
        assert(pc == make_piece(us, KING));
        assert(captured == make_piece(us, ROOK));

        Square rfrom, rto;
        do_castling<true>(us, from, to, rfrom, rto, &dts, &dp);

        // Castling with a zombie rook turns it back into a normal piece
        if (st->zombieBB & rfrom)
        {
            st->zombieBB ^= rfrom;
            k ^= Zobrist::zombie[rfrom];
        }

        k ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
        st->nonPawnKey[us] ^= Zobrist::psq[captured][rfrom] ^ Zobrist::psq[captured][rto];
        captured = NO_PIECE;
    }
    else if (captured)
    {
        Square capsq = to;

        // If the captured piece is a pawn, update pawn hash key, otherwise
        // update non-pawn material.
        if (type_of(captured) == PAWN)
        {
            if (m.type_of() == EN_PASSANT)
            {
                capsq -= pawn_push(us);

                assert(pc == make_piece(us, PAWN));
                assert(to == st->epSquare);
                assert(relative_rank(us, to) == RANK_6);
                assert(piece_on(to) == NO_PIECE);
                assert(piece_on(capsq) == make_piece(them, PAWN));

                // Update board and piece lists in ep case, normal captures are updated later
                remove_piece(capsq, &dts);
            }

            st->pawnKey ^= Zobrist::psq[captured][capsq];
        }
        else
        {
            st->nonPawnMaterial[them] -= PieceValue[captured];
            st->nonPawnKey[them] ^= Zobrist::psq[captured][capsq];

            if (type_of(captured) <= BISHOP)
                st->minorPieceKey ^= Zobrist::psq[captured][capsq];
        }

        dp.remove_pc = captured;
        dp.remove_sq = capsq;

        k ^= Zobrist::psq[captured][capsq];
        st->materialKey ^=
          Zobrist::psq[captured][8 + pieceCount[captured] - (m.type_of() != EN_PASSANT)];

        // Reset rule 50 counter
        st->rule50 = 0;
    }
    else
        dp.remove_sq = SQ_NONE;

    // Update hash key
    k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

    // Reset en passant square
    if (st->epSquare != SQ_NONE)
    {
        k ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    // Update castling rights.
    k ^= Zobrist::castling[st->castlingRights];
    st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[to]);
    k ^= Zobrist::castling[st->castlingRights];

    // If the moving piece is a pawn do some special extra work
    if (type_of(pc) == PAWN)
    {
        // Check if the en passant square needs to be set. Accurate e.p. info is needed
        // for correct zobrist key generation and 3-fold checking.
        if ((int(to) ^ int(from)) == 16)
        {
            Square   epSquare = to - pawn_push(us);
            Bitboard pawns    = attacks_bb(PAWN, epSquare, us) & pieces(them, PAWN);

            // If there are no pawns attacking the ep square, ep is not possible.
            // Nor with the king in check by SCP-096.
            if (pawns && !(st->scpKingForbidden & pieces(them, KING)))
            {
                Square   ksq         = square<KING>(them);
                Bitboard notBlockers = ~st->previous->blockersForKing[them];
                bool     noDiscovery = (from & notBlockers) || file_of(from) == file_of(ksq);

                // If the pawn gives discovered check, ep is never legal. Else, if at least one
                // pawn was not a blocker for the enemy king or lies on the same line as the
                // enemy king and en passant square, a legal capture exists.
                // Under SCP-049 any non-zombie pawn may capture
                if (scpType == SCP_049 ? bool(pawns & ~st->zombieBB)
                                       : noDiscovery && (pawns & (notBlockers | line_bb(epSquare, ksq))))
                {
                    st->epSquare = epSquare;
                    k ^= Zobrist::enpassant[file_of(epSquare)];
                }
            }
        }

        else if (m.type_of() == PROMOTION)
        {
            PieceType pt        = m.promotion_type();
            Piece     promotion = make_piece(us, pt);

            assert(relative_rank(us, to) == RANK_8);
            assert(pt >= KNIGHT && pt <= QUEEN);

            dp.add_pc = promotion;
            dp.add_sq = to;
            dp.to     = SQ_NONE;

            // Update hash keys
            // Zobrist::psq[pc][to] is zero, so we don't need to clear it
            k ^= Zobrist::psq[promotion][to];
            st->materialKey ^= Zobrist::psq[promotion][8 + pieceCount[promotion]]
                             ^ Zobrist::psq[pc][8 + pieceCount[pc] - 1];
            st->nonPawnKey[us] ^= Zobrist::psq[promotion][to];

            if (pt <= BISHOP)
                st->minorPieceKey ^= Zobrist::psq[promotion][to];

            // Update material
            st->nonPawnMaterial[us] += PieceValue[promotion];
        }

        // Update pawn hash key
        st->pawnKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

        // Reset rule 50 draw counter
        st->rule50 = 0;
    }

    else
    {
        st->nonPawnKey[us] ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

        if (type_of(pc) <= BISHOP)
            st->minorPieceKey ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];
    }

    // Update the key with the final value
    st->key = k;

    if (history)
    {
        prefetch(&history->pawn_entry(*this)[pc][to]);
        prefetch(&history->pawn_correction_entry(*this));
        prefetch(&history->minor_piece_correction_entry(*this));
        prefetch(&history->nonpawn_correction_entry<WHITE>(*this));
        prefetch(&history->nonpawn_correction_entry<BLACK>(*this));
    }

    // Move the piece. The tricky Chess960 castling is handled earlier
    if (m.type_of() != CASTLING)
    {
        Piece toPc = pc;
        if (m.type_of() == PROMOTION)
            toPc = make_piece(us, m.promotion_type());

        if (captured && m.type_of() != EN_PASSANT)
        {
            remove_piece(from, &dts);
            swap_piece(to, toPc, &dts);
        }
        else if (pc == toPc)
            move_piece(from, to, &dts);
        else
        {
            remove_piece(from, &dts);
            put_piece(toPc, to, &dts);
        }
    }

    // Set capture piece
    st->capturedPiece = captured;

    // Calculate checkers bitboard (if move gives check)
    st->checkersBB = givesCheck ? attackers_to(square<KING>(them)) & pieces(us) : 0;

    if (st->scpKingForbidden & pieces(them, KING))
        st->checkersBB |= st->scpBB;

    sideToMove = ~sideToMove;

    // Update king attacks used for fast check detection
    set_extra_bitboards();

    // Both SCPs act after Black's move
    st->scpOutcome   = SCP_NONE;
    st->scpCaptured  = NO_PIECE;
    st->phasePending = false;
    if (us == BLACK && scpType == SCP_096)
        scp_act(dp);

    // SCP-049's turn is random and is played separately by do_scp049_phase()
    if (scpType == SCP_049)
    {
        if (us == BLACK)
        {
            st->phasePending = true;
            st->key ^= Zobrist::phasePending;
        }
        scp049_check_info();
    }

    // Calculate the repetition info. It is the ply distance from the previous
    // occurrence of the same position, negative in the 3-fold case, or zero
    // if the position was not repeated.
    st->repetition = 0;
    int end        = std::min(st->rule50, st->pliesFromNull);
    if (end >= 4)
    {
        StateInfo* stp = st->previous->previous;
        for (int i = 4; i <= end; i += 2)
        {
            stp = stp->previous->previous;
            if (stp->key == st->key)
            {
                st->repetition = stp->repetition ? -i : i;
                break;
            }
        }
    }

    assert(pos_is_ok());

    dpps.after[WHITE] = pieces(WHITE, PAWN);
    dpps.after[BLACK] = pieces(BLACK, PAWN);

    assert(dp.pc != NO_PIECE);
    assert(!(bool(captured) || m.type_of() == CASTLING) ^ (dp.remove_sq != SQ_NONE));
    assert(dp.from != SQ_NONE);
    assert(!(dp.add_sq != SQ_NONE) ^ (m.type_of() == PROMOTION || m.type_of() == CASTLING));
}


// Unmakes a move. When it returns, the position should
// be restored to exactly the same state as before the move was made.
void Position::undo_move(Move m) {

    assert(m.is_ok());

    sideToMove = ~sideToMove;

    // Restore the piece SCP-096 took first, as it may be the one that moved
    if (st->scpCaptured)
        put_piece(st->scpCaptured, st->scpCapturedSq);

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = piece_on(to);

    assert(empty(from) || m.type_of() == CASTLING);
    assert(type_of(st->capturedPiece) != KING);

    if (m.type_of() == PROMOTION)
    {
        assert(relative_rank(us, to) == RANK_8);
        assert(type_of(pc) == m.promotion_type());
        assert(type_of(pc) >= KNIGHT && type_of(pc) <= QUEEN);

        pc = make_piece(us, PAWN);
        swap_piece(to, pc);
    }

    if (m.type_of() == CASTLING)
    {
        Square rfrom, rto;
        do_castling<false>(us, from, to, rfrom, rto);
    }
    else
    {
        move_piece(to, from);  // Put the piece back at the source square

        if (st->capturedPiece)
        {
            Square capsq = to;

            if (m.type_of() == EN_PASSANT)
            {
                capsq -= pawn_push(us);

                assert(type_of(pc) == PAWN);
                assert(to == st->previous->epSquare);
                assert(relative_rank(us, to) == RANK_6);
                assert(piece_on(capsq) == NO_PIECE);
                assert(st->capturedPiece == make_piece(~us, PAWN));
            }

            put_piece(st->capturedPiece, capsq);  // Restore the captured piece
        }
    }

    // Finally point our state pointer back to the previous state
    st = st->previous;
    --gamePly;

    assert(pos_is_ok());
}

inline void add_dirty_threat(DirtyThreats* const dts,
                             bool                putPiece,
                             Piece               pc,
                             Piece               threatened,
                             Square              s,
                             Square              threatenedSq) {
    dts->list.push_back({pc, threatened, s, threatenedSq, putPiece});
}

#ifdef USE_AVX512ICL
// Given a DirtyThreat template and bit offsets to insert the piece type and square, write the threats
// present at the given bitboard.
template<int SqShift, int PcShift>
void write_multiple_dirties(const Position& p,
                            Bitboard        mask,
                            DirtyThreat     dt_template,
                            DirtyThreats*   dts) {
    static_assert(sizeof(DirtyThreat) == 4);

    const __m512i board    = _mm512_loadu_si512(p.piece_array().data());
    const int     dt_count = popcount(mask);
    assert(dt_count <= 16);

    const __m512i template_v = _mm512_set1_epi32(dt_template.raw());
    auto*         write      = dts->list.make_space(dt_count);

    // Extract the list of squares and upconvert to 32 bits. There are never more than 16
    // incoming threats so this is sufficient.
    __m512i threat_squares = _mm512_maskz_compress_epi8(mask, AllSquares);
    threat_squares         = _mm512_cvtepi8_epi32(_mm512_castsi512_si128(threat_squares));

    __m512i threat_pieces =
      _mm512_maskz_permutexvar_epi8(0x1111111111111111ULL, threat_squares, board);

    // Shift the piece and square into place
    threat_squares = _mm512_slli_epi32(threat_squares, SqShift);
    threat_pieces  = _mm512_slli_epi32(threat_pieces, PcShift);

    const __m512i dirties =
      _mm512_ternarylogic_epi32(template_v, threat_squares, threat_pieces, 254 /* A | B | C */);
    _mm512_storeu_si512(write, dirties);
}
#endif

constexpr bool can_slider_threat(Piece pc, Piece slider) {
    return type_of(pc) != QUEEN || type_of(slider) == QUEEN;
}

template<bool ComputeRay>
void Position::update_piece_threats(Piece               pc,
                                    bool                putPiece,
                                    Square              s,
                                    DirtyThreats* const dts,
                                    // Silence spurious warning on GCC 10
                                    [[maybe_unused]] Bitboard noRaysContaining) const {
    const Bitboard occupied         = pieces();
    const auto [bAttacks, rAttacks] = both_attacks_bb(s, occupied);
    const Bitboard  sliderAttacks   = bAttacks | rAttacks;
    const Bitboard  occupiedNoK     = occupied ^ pieces(KING);
    const PieceType pt              = type_of(pc);
    const Bitboard  sliders = (pieces(BISHOP, QUEEN) & bAttacks) | (pieces(ROOK, QUEEN) & rAttacks);

    auto process_sliders = [&](bool addDirectAttacks) {
        Bitboard b = sliders;
        while (b)
        {
            Square sliderSq = pop_lsb(b);
            Piece  slider   = piece_on(sliderSq);

            const Bitboard ray        = ray_pass_bb(sliderSq, s);
            const Bitboard discovered = ray & sliderAttacks & occupiedNoK;

            assert(!more_than_one(discovered));
            if (discovered && (ray & noRaysContaining) != noRaysContaining)
            {
                const Square threatenedSq = lsb(discovered);
                const Piece  threatenedPc = piece_on(threatenedSq);
                if (can_slider_threat(threatenedPc, slider))
                    add_dirty_threat(dts, !putPiece, slider, threatenedPc, sliderSq, threatenedSq);
            }

            if (addDirectAttacks && can_slider_threat(pc, slider))
                add_dirty_threat(dts, putPiece, slider, pc, sliderSq, s);
        }
    };

    // Kings emit no direct threats
    if (pt == KING)
    {
        if constexpr (ComputeRay)
            process_sliders(false);
        return;
    }

    const Bitboard threatTargets = pt == PAWN                 ? pieces(KNIGHT, ROOK)
                                 : pt == BISHOP || pt == ROOK ? pieces(PAWN, KNIGHT, BISHOP, ROOK)
                                                              : occupiedNoK;
    Bitboard       threatened    = (pt == BISHOP  ? bAttacks
                                    : pt == ROOK  ? rAttacks
                                    : pt == QUEEN ? sliderAttacks
                                    : pt == PAWN  ? PseudoAttacks[color_of(pc)][s]
                                                  : PseudoAttacks[pt][s])
                        & threatTargets;
    Bitboard incomingThreats = PseudoAttacks[KNIGHT][s] & pieces(KNIGHT);

    if (pt == KNIGHT || pt == ROOK)
        incomingThreats |= (attacks_bb(PAWN, s, WHITE) & pieces(BLACK, PAWN))
                         | (attacks_bb(PAWN, s, BLACK) & pieces(WHITE, PAWN));

#ifdef USE_AVX512ICL
    write_multiple_dirties<DirtyThreat::ThreatenedSqOffset, DirtyThreat::ThreatenedPcOffset>(
      *this, threatened, {pc, NO_PIECE, s, Square(0), putPiece}, dts);

    const Bitboard directSliders = pt == QUEEN ? sliders & pieces(QUEEN) : sliders;
    write_multiple_dirties<DirtyThreat::PcSqOffset, DirtyThreat::PcOffset>(
      *this, directSliders | incomingThreats, {NO_PIECE, pc, Square(0), s, putPiece}, dts);

    // For ICL, direct threats were written above
    if constexpr (ComputeRay)
        process_sliders(false);
#else
    while (threatened)
    {
        Square threatenedSq = pop_lsb(threatened);
        Piece  threatenedPc = piece_on(threatenedSq);
        assert(threatenedSq != s);
        assert(threatenedPc != NO_PIECE);

        add_dirty_threat(dts, putPiece, pc, threatenedPc, s, threatenedSq);
    }

    if constexpr (ComputeRay)
        process_sliders(true);
    else
        incomingThreats |= pt == QUEEN ? sliders & pieces(QUEEN) : sliders;

    while (incomingThreats)
    {
        Square srcSq = pop_lsb(incomingThreats);
        Piece  srcPc = piece_on(srcSq);
        assert(srcSq != s);
        assert(srcPc != NO_PIECE);

        add_dirty_threat(dts, putPiece, srcPc, pc, srcSq, s);
    }
#endif
}

Key Position::prefetch_key(Move m) const {
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = piece_on(from);
    Piece  captured = piece_on(to);
    Key    k        = st->key ^ Zobrist::side;

    k ^= Zobrist::psq[captured][to] ^ Zobrist::psq[pc][to] ^ Zobrist::psq[pc][from];

    if (captured || type_of(pc) == PAWN)
        return k;

    return adjust_key50<true>(k);
}

// Helper used to do/undo a castling move. This is a bit
// tricky in Chess960 where from/to squares can overlap.
template<bool Do>
void Position::do_castling(Color               us,
                           Square              from,
                           Square&             to,
                           Square&             rfrom,
                           Square&             rto,
                           DirtyThreats* const dts,
                           DirtyPiece* const   dp) {

    bool kingSide = to > from;
    rfrom         = to;  // Castling is encoded as "king captures friendly rook"
    rto           = relative_square(us, kingSide ? SQ_F1 : SQ_D1);
    to            = relative_square(us, kingSide ? SQ_G1 : SQ_C1);

    assert(!Do || dp);

    if (Do)
    {
        dp->to        = to;
        dp->remove_pc = dp->add_pc = make_piece(us, ROOK);
        dp->remove_sq              = rfrom;
        dp->add_sq                 = rto;
    }

    // Remove both pieces first since squares could overlap in Chess960
    remove_piece(Do ? from : to, dts);
    remove_piece(Do ? rfrom : rto, dts);
    put_piece(make_piece(us, KING), Do ? to : from, dts);
    put_piece(make_piece(us, ROOK), Do ? rto : rfrom, dts);
}


// Used to do a "null move": it flips
// the side to move without executing any move on the board.
void Position::do_null_move(StateInfo& newSt) {

    assert(!checkers());
    assert(&newSt != st);

    std::memcpy(&newSt, st, sizeof(StateInfo));

    newSt.previous = st;
    st             = &newSt;

    if (st->epSquare != SQ_NONE)
    {
        st->key ^= Zobrist::enpassant[file_of(st->epSquare)];
        st->epSquare = SQ_NONE;
    }

    st->key ^= Zobrist::side;

    st->pliesFromNull = 0;

    st->capturedPiece = NO_PIECE;
    st->scpCaptured   = NO_PIECE;
    st->scpOutcome    = SCP_NONE;
    st->phasePending  = false;

    sideToMove = ~sideToMove;

    set_extra_bitboards();

    st->repetition = 0;

    assert(pos_is_ok());
}


// Must be used to undo a "null move"
void Position::undo_null_move() {

    assert(!checkers());

    st         = st->previous;
    sideToMove = ~sideToMove;
}


// Tests if the SEE (Static Exchange Evaluation)
// value of the move is greater or equal to the given threshold. We'll use an
// algorithm similar to alpha-beta pruning with a null window.
bool Position::see_ge(Move m, int threshold) const {

    assert(m.is_ok());

    // Only deal with normal moves, assume others pass a simple SEE
    if (m.type_of() != NORMAL)
        return VALUE_ZERO >= threshold;

    Square from = m.from_sq(), to = m.to_sq();

    assert(piece_on(from) != NO_PIECE);

    int swap = PieceValue[piece_on(to)] - threshold;
    if (swap < 0)
        return false;

    swap = PieceValue[piece_on(from)] - swap;
    if (swap <= 0)
        return true;

    assert(color_of(piece_on(from)) == sideToMove);
    Bitboard occupied  = blockers() ^ from ^ to;  // xoring to is important for pinned piece logic
    Color    stm       = sideToMove;
    Bitboard attackers = attackers_to(to, occupied);
    Bitboard stmAttackers, bb;
    int      res = 1;

    while (true)
    {
        stm = ~stm;
        attackers &= occupied & ~st->zombieBB;  // zombies do not capture

        // If stm has no more attackers then give up: stm loses
        if (!(stmAttackers = attackers & pieces(stm)))
            break;

        // Don't allow pinned pieces to attack as long as there are
        // pinners on their original square.
        if (pinners(~stm) & occupied)
        {
            stmAttackers &= ~blockers_for_king(stm);

            if (!stmAttackers)
                break;
        }

        res ^= 1;

        // Locate and remove the next least valuable attacker, and add to
        // the bitboard 'attackers' any X-ray attackers behind it.
        if ((bb = stmAttackers & pieces(PAWN)))
        {
            if ((swap = PawnValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb(BISHOP, to, occupied) & pieces(BISHOP, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(KNIGHT)))
        {
            if ((swap = KnightValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);
        }

        else if ((bb = stmAttackers & pieces(BISHOP)))
        {
            if ((swap = BishopValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb(BISHOP, to, occupied) & pieces(BISHOP, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(ROOK)))
        {
            if ((swap = RookValue - swap) < res)
                break;
            occupied ^= least_significant_square_bb(bb);

            attackers |= attacks_bb(ROOK, to, occupied) & pieces(ROOK, QUEEN);
        }

        else if ((bb = stmAttackers & pieces(QUEEN)))
        {
            swap = QueenValue - swap;
            //  implies that the previous recapture was done by a higher rated piece than a Queen (King is excluded)
            assert(swap >= res);
            occupied ^= least_significant_square_bb(bb);

            const auto [bishopAttacks, rookAttacks] = both_attacks_bb(to, occupied);
            attackers |=
              (bishopAttacks & pieces(BISHOP, QUEEN)) | (rookAttacks & pieces(ROOK, QUEEN));
        }

        else  // KING
              // If we "capture" with the king but the opponent still has attackers,
              // reverse the result.
            return (attackers & ~pieces(stm)) ? res ^ 1 : res;
    }

    return bool(res);
}

// Tests whether the position is drawn by 50-move rule
// or by repetition. It does not detect stalemates.
bool Position::is_draw(int ply) const {

    // SCP-049 has no draws
    if (scpType == SCP_049)
        return false;

    if (st->rule50 > 99 && (!checkers() || MoveList<ALL>(*this).size()))
        return true;

    return is_repetition(ply);
}

// Return a draw score if a position repeats once earlier but strictly
// after the root, or repeats twice before or at the root.
bool Position::is_repetition(int ply) const { return st->repetition && st->repetition < ply; }

// Tests whether there has been at least one repetition
// of positions since the last capture or pawn move.
bool Position::has_repeated() const {

    StateInfo* stc = st;
    int        end = std::min(st->rule50, st->pliesFromNull);
    while (end-- >= 4)
    {
        if (stc->repetition)
            return true;

        stc = stc->previous;
    }
    return false;
}


// Tests if the position has a move which draws by repetition.
// This function accurately matches the outcome of is_draw() over all legal moves.
bool Position::upcoming_repetition(int ply) const {

    if (scpType == SCP_049)
        return false;

    int end = std::min(st->rule50, st->pliesFromNull);

    if (end < 3)
        return false;

    Key        originalKey = st->key;
    StateInfo* stp         = st->previous;
    Key        other       = originalKey ^ stp->key ^ Zobrist::side;

    for (int i = 3; i <= end; i += 2)
    {
        stp = stp->previous;
        other ^= stp->key ^ stp->previous->key ^ Zobrist::side;
        stp = stp->previous;

        if (other != 0)
            continue;

        Key moveKey = originalKey ^ stp->key;

        if (int j = H1(moveKey); cuckoo[j] == moveKey || (j = H2(moveKey), cuckoo[j] == moveKey))
        {
            Move   move = cuckooMove[j];
            Square s1   = move.from_sq();
            Square s2   = move.to_sq();

            if (!((between_bb(s1, s2) ^ s2) & blockers()))
            {
                if (ply > i)
                    return true;

                // For nodes before or at the root, check that the move is a
                // repetition rather than a move to the current position.
                if (stp->repetition)
                    return true;
            }
        }
    }
    return false;
}


// Places the SCP on s (SQ_NONE removes it). The hash key is not updated.
void Position::set_scp(Square s) {
    st->scpBB            = s == SQ_NONE ? 0 : square_bb(s);
    st->scpKingForbidden =
      s == SQ_NONE || scpType != SCP_096 ? 0 : square_bb(s) | attacks_bb(KNIGHT, s);
}


// Returns the square of the piece SCP-096 would capture, or SQ_NONE: the attacker
// closest by Euclidean distance, ties broken by higher rank, then lower file.
Square Position::scp_victim() const {

    if (!st->scpBB || scpType != SCP_096)
        return SQ_NONE;

    const Square s         = lsb(st->scpBB);
    Bitboard     attackers = attackers_to(s, pieces());
    Square       victim    = SQ_NONE;
    int          best      = INT_MAX;

    while (attackers)
    {
        const Square a   = pop_lsb(attackers);
        const int    df  = int(file_of(a)) - int(file_of(s));
        const int    dr  = int(rank_of(a)) - int(rank_of(s));
        const int    ord = (df * df + dr * dr) * 64 + (RANK_8 - rank_of(a)) * 8 + file_of(a);

        if (ord < best)
            best = ord, victim = a;
    }

    return victim;
}


// SCP-096's turn, played by do_move() after Black's move: the SCP captures the
// nearest piece attacking it and moves to its square.
void Position::scp_act(DirtyPiece& dp) {

    assert(sideToMove == WHITE);

    const Square victimSq = scp_victim();

    if (victimSq == SQ_NONE)
        return;

    // Mate and stalemate are decided before the SCP acts
    if (!MoveList<ALL>(*this).size())
        return;

    const Piece victim = piece_on(victimSq);
    const Color vc     = color_of(victim);

    if (type_of(victim) == KING)
    {
        // The king stays on the board; search scores the node via scp_outcome()
        st->scpOutcome = vc == sideToMove ? SCP_LOSS : SCP_WIN;
        st->key ^= Zobrist::scpOutcome;
        return;
    }

    Key k = st->key;

    if (type_of(victim) == PAWN)
        st->pawnKey ^= Zobrist::psq[victim][victimSq];
    else
    {
        st->nonPawnMaterial[vc] -= PieceValue[victim];
        st->nonPawnKey[vc] ^= Zobrist::psq[victim][victimSq];

        if (type_of(victim) <= BISHOP)
            st->minorPieceKey ^= Zobrist::psq[victim][victimSq];
    }

    k ^= Zobrist::psq[victim][victimSq];
    st->materialKey ^= Zobrist::psq[victim][8 + pieceCount[victim] - 1];

    // A rook taken on its corner loses its castling right
    k ^= Zobrist::castling[st->castlingRights];
    st->castlingRights &= ~castlingRightsMask[victimSq];
    k ^= Zobrist::castling[st->castlingRights];

    remove_piece(victimSq);
    st->scpCaptured   = victim;
    st->scpCapturedSq = victimSq;

    k ^= Zobrist::scp[lsb(st->scpBB)] ^ Zobrist::scp[victimSq];
    set_scp(victimSq);

    st->key = k;

    // No NNUE diff for this capture: force a refresh
    dp.scpCapture = true;

    // The SCP may have opened or closed lines
    const Square ksq = square<KING>(sideToMove);
    st->checkersBB   = attackers_to(ksq) & pieces(~sideToMove);
    if (st->scpKingForbidden & ksq)
        st->checkersBB |= st->scpBB;

    set_extra_bitboards();
    validate_ep_after_scp();

    // White can capture an exposed Black king
    if (attackers_to(square<KING>(~sideToMove)) & pieces(sideToMove))
    {
        st->scpOutcome = SCP_WIN;
        st->key ^= Zobrist::scpOutcome;
    }
}


// Drops the en passant square if SCP-096's capture made it illegal
void Position::validate_ep_after_scp() {

    if (st->epSquare == SQ_NONE)
        return;

    const Color  us     = sideToMove;
    const Square epSq   = st->epSquare;
    const Square ksq    = square<KING>(us);
    Bitboard     pawns  = attacks_bb(PAWN, epSq, ~us) & pieces(us, PAWN);
    Bitboard     target = pieces(~us, PAWN) & (epSq + pawn_push(~us));
    Bitboard     occ    = blockers() ^ target ^ epSq;
    bool         legal  = false;

    if (target && !(st->scpKingForbidden & ksq))
        while (pawns)
            legal |= !(attackers_to(ksq, occ ^ pop_lsb(pawns)) & pieces(~us) & ~target);

    if (!legal)
    {
        st->key ^= Zobrist::enpassant[file_of(epSq)];
        st->epSquare = SQ_NONE;
    }
}


// SCP-049 (iwantcheckmate.com/challenges/scp-049). There is no check and a game
// ends when a king is captured. After each Black move the SCP steps to a random
// empty orthogonally adjacent square and turns the pieces around it into zombies,
// then each older zombie makes a random move. Zombies cannot be moved or captured,
// and capture any piece except zombies and the king of their own color.

// Returns true if the side to move can capture the enemy king
bool Position::can_capture_king() const {

    const Square ksq = square<KING>(~sideToMove);

    return !(st->zombieBB & ksq)
        && (attackers_to(ksq, blockers()) & pieces(sideToMove) & ~st->zombieBB);
}


// Sets the checkers (enemy pieces able to capture our king, which search treats
// as a check) and the outcome when the side to move can capture the enemy king.
void Position::scp049_check_info() const {

    const Square ksq = square<KING>(sideToMove);

    st->checkersBB = (st->zombieBB & ksq)
                     ? 0
                     : attackers_to(ksq, blockers()) & pieces(~sideToMove) & ~st->zombieBB;

    // Undecided while the SCP still has to act: it may zombify the attacker
    if (!st->phasePending && can_capture_king())
    {
        st->scpOutcome = SCP_WIN;
        st->key ^= Zobrist::scpOutcome;
    }
}


// The site's attack test, used only for castling: the SCP does not block lines,
// and for White attacks every zombie counts, zombie pawns in both directions.
bool Position::scp049_attacked(Square s, Color by) const {

    const Bitboard occ     = pieces();
    const Bitboard zombies = by == WHITE ? st->zombieBB : Bitboard(0);
    const Bitboard real    = pieces(by) & ~st->zombieBB;
    const Bitboard att     = real | zombies;

    const auto [bishopAttacks, rookAttacks] = both_attacks_bb(s, occ);

    return (rookAttacks & att & pieces(ROOK, QUEEN)) | (bishopAttacks & att & pieces(BISHOP, QUEEN))
         | (attacks_bb(KNIGHT, s) & att & pieces(KNIGHT)) | (attacks_bb(KING, s) & att & pieces(KING))
         | (attacks_bb(PAWN, s, ~by) & real & pieces(PAWN))
         | ((attacks_bb(PAWN, s, WHITE) | attacks_bb(PAWN, s, BLACK)) & zombies & pieces(PAWN));
}


// Empty squares SCP-049 may step to, in the site's order: up, right, down, left
int Position::scp049_destinations(Square dests[4]) const {

    const Square s = lsb(st->scpBB);
    int          n = 0;

    const bool canUp = rank_of(s) < RANK_8, canDown = rank_of(s) > RANK_1;
    const bool canRight = file_of(s) < FILE_H, canLeft = file_of(s) > FILE_A;

    for (auto [ok, d] : {std::pair{canUp, NORTH}, std::pair{canRight, EAST},
                         std::pair{canDown, SOUTH}, std::pair{canLeft, WEST}})
        if (ok && empty(s + d))
            dests[n++] = s + d;

    return n;
}


// Returns the squares the zombie on 'from' may move to
Bitboard Position::scp049_zombie_moves(Square from) const {

    const Piece    pc       = piece_on(from);
    const Color    c        = color_of(pc);
    const Bitboard occ      = pieces() | st->scpBB;
    const Bitboard allowed  = ~st->scpBB & ~st->zombieBB & ~pieces(c, KING);
    Bitboard       targets  = 0;

    switch (type_of(pc))
    {
    case PAWN : {
        // Zombie pawns move and capture in both directions
        for (Direction d : {NORTH, SOUTH})
        {
            if ((d == NORTH && rank_of(from) == RANK_8) || (d == SOUTH && rank_of(from) == RANK_1))
                continue;
            if (!(occ & (from + d)))
                targets |= from + d;
        }
        targets |= (attacks_bb(PAWN, from, WHITE) | attacks_bb(PAWN, from, BLACK)) & pieces();
        break;
    }
    case KNIGHT :
    case KING :
        targets = attacks_bb(type_of(pc), from);
        break;
    default :
        targets = attacks_bb(type_of(pc), from, occ);
    }

    return targets & allowed;
}


// Plays SCP-049's turn. 'dir' indexes scp049_destinations(), -1 picks at random.
// In search mode zombies never capture a king: the probability that one would is
// added to 'res' and a non-capturing move is drawn instead.
void Position::do_scp049_phase(int                       dir,
                               u64                       seed,
                               bool                      searchMode,
                               StateInfo&                newSt,
                               ScpPhaseUndo&             undo,
                               DirtyPiece&               dp,
                               ScpPhaseResult&           res,
                               std::vector<std::string>* events) {

    assert(scpType == SCP_049 && st->phasePending && sideToMove == WHITE);

    std::memcpy(&newSt, st, sizeof(StateInfo));
    newSt.previous = st;
    st             = &newSt;

    st->phasePending = false;
    st->scpOutcome   = SCP_NONE;
    st->scpCaptured  = NO_PIECE;

    std::copy(std::begin(board), std::end(board), undo.board.begin());
    undo.byTypeBB  = byTypeBB;
    undo.byColorBB = byColorBB;
    std::copy(std::begin(pieceCount), std::end(pieceCount), undo.pieceCount.begin());

    // Force an accumulator refresh
    dp            = DirtyPiece{};
    dp.pc         = NO_PIECE;
    dp.from = dp.to = dp.remove_sq = dp.add_sq = SQ_NONE;
    dp.scpCapture = true;

    PRNG rng(seed | 1);
    auto say = [&](const std::string& e) {
        if (events)
            events->push_back(e);
    };
    auto sq = [](Square s) { return UCIEngine::square(s); };

    // 1. Step
    Square dests[4];
    if (const int n = scp049_destinations(dests))
    {
        const Square from = lsb(st->scpBB);
        const Square to   = dests[dir >= 0 && dir < n ? dir : int(rng.rand<u64>() % u64(n))];
        set_scp(to);
        say("move " + sq(from) + " " + sq(to));
    }

    // 2. Zombify the adjacent pieces
    const Bitboard oldZombies = st->zombieBB;
    const Bitboard cured      = attacks_bb(KING, lsb(st->scpBB)) & pieces() & ~st->zombieBB;
    st->zombieBB |= cured;
    for (Bitboard b = cured; b;)
        say("zombify " + sq(pop_lsb(b)));

    // 3. Older zombies move
    Color kingTaken = COLOR_NB;
    for (Bitboard movers = oldZombies; movers && res.survive > 0;)
    {
        const Square from    = pop_lsb(movers);
        Bitboard     targets = scp049_zombie_moves(from);

        if (!targets)
            continue;

        int n = popcount(targets);

        if (searchMode)
        {
            if (const Bitboard k = targets & pieces(KING))
            {
                // Only the king of the other color can be a target
                const double p = res.survive * popcount(k) / n;
                res.pKingLoss[color_of(piece_on(lsb(k)))] += p;
                res.survive -= p;
                targets &= ~k;
                n -= popcount(k);
                if (!n)
                {
                    res.survive = 0;  // every move captures a king
                    break;
                }
            }
        }

        // Select the i-th target
        int i = int(rng.rand<u64>() % u64(n));
        while (i--)
            targets &= targets - 1;
        const Square to = lsb(targets);

        const Piece pc       = piece_on(from);
        const Piece captured = piece_on(to);

        if (type_of(captured) == KING)
        {
            kingTaken = color_of(captured);
            say("zombie " + sq(from) + " " + sq(to) + " king");
            break;
        }

        if (captured)
            remove_piece(to);

        move_piece(from, to);
        st->zombieBB ^= from | to;
        st->castlingRights &= ~(castlingRightsMask[from] | castlingRightsMask[to]);

        // Zombie pawns promote to a queen on either back rank
        const bool promotes = type_of(pc) == PAWN && (rank_of(to) == RANK_1 || rank_of(to) == RANK_8);
        if (promotes)
        {
            remove_piece(to);
            put_piece(make_piece(color_of(pc), QUEEN), to);
        }

        say("zombie " + sq(from) + " " + sq(to) + (captured ? " x" : "") + (promotes ? " =q" : ""));
    }

    if (searchMode && res.survive <= 0)
        res.survive = 0;

    scp049_after_change();

    // Outside search, a zombie captured a king
    if (kingTaken != COLOR_NB)
    {
        st->scpOutcome = kingTaken == sideToMove ? SCP_LOSS : SCP_WIN;
        st->key ^= Zobrist::scpOutcome;
    }
}


void Position::skip_scp049_phase(StateInfo& newSt) {

    assert(scpType == SCP_049 && st->phasePending);

    std::memcpy(&newSt, st, sizeof(StateInfo));
    newSt.previous = st;
    st             = &newSt;

    st->phasePending = false;
    st->key ^= Zobrist::phasePending;
    scp049_check_info();
}


void Position::undo_scp049_phase(const ScpPhaseUndo& undo) {

    std::copy(undo.board.begin(), undo.board.end(), std::begin(board));
    byTypeBB  = undo.byTypeBB;
    byColorBB = undo.byColorBB;
    std::copy(undo.pieceCount.begin(), undo.pieceCount.end(), std::begin(pieceCount));

    st = st->previous;
}


// Recomputes keys, material and check info after SCP-049's turn
void Position::scp049_after_change() {

    // Drop en passant if the pawn was captured or zombified, or the square taken
    if (Square ep = st->epSquare; ep != SQ_NONE)
    {
        const Bitboard target = pieces(~sideToMove, PAWN) & ~st->zombieBB & (ep - pawn_push(sideToMove));
        const Bitboard pawns  = attacks_bb(PAWN, ep, ~sideToMove) & pieces(sideToMove, PAWN) & ~st->zombieBB;
        if (!target || !pawns || !empty(ep) || (st->scpBB & ep))
            st->epSquare = SQ_NONE;
    }

    const int        rule50     = st->rule50;
    const Bitboard   zombieBB   = st->zombieBB;
    const Bitboard   scpBB      = st->scpBB;

    set_state();

    assert(st->rule50 == rule50 && st->zombieBB == zombieBB && st->scpBB == scpBB);
    (void) rule50, (void) zombieBB, (void) scpBB;
}


// Flips position with the white and black sides reversed. This
// is only useful for debugging e.g. for finding evaluation symmetry bugs.
std::optional<PositionSetError> Position::flip() {

    string            f, token;
    std::stringstream ss(fen());

    for (Rank r = RANK_8;; --r)  // Piece placement
    {
        std::getline(ss, token, r > RANK_1 ? '/' : ' ');
        f.insert(0, token + (f.empty() ? " " : "/"));

        if (r == RANK_1)
            break;
    }

    ss >> token;                        // Active color
    f += (token == "w" ? "B " : "W ");  // Will be lowercased later

    ss >> token;  // Castling availability
    f += token + " ";

    std::transform(f.begin(), f.end(), f.begin(),
                   [](unsigned char c) { return char(islower(c) ? toupper(c) : tolower(c)); });

    ss >> token;  // En passant square
    f += (token == "-" ? token : token.replace(1, 1, token[1] == '3' ? "6" : "3"));

    std::getline(ss, token);  // Half and full moves, SCP fields
    if (auto i = token.find("scp"); i != string::npos)
        for (usize j = i; j < token.size(); ++j)  // flip SCP and zombie ranks
            if (token[j] >= '1' && token[j] <= '8' && j > 0 && token[j - 1] >= 'a'
                && token[j - 1] <= 'h')
                token[j] = char('1' + '8' - token[j]);
    f += token;

    return set(f, is_chess960(), st);
}


bool Position::material_key_is_ok() const { return compute_material_key() == st->materialKey; }


// Performs some consistency checks for the position object
// and raise an assert if something wrong is detected.
// This is meant to be helpful when debugging.
bool Position::pos_is_ok() const {

    if ((sideToMove != WHITE && sideToMove != BLACK) || piece_on(square<KING>(WHITE)) != W_KING
        || piece_on(square<KING>(BLACK)) != B_KING
        || (ep_square() != SQ_NONE && relative_rank(sideToMove, ep_square()) != RANK_6))
        assert(0 && "pos_is_ok: Default");

    if (count<KING>(WHITE) != 1 || count<KING>(BLACK) != 1
        || (!st->scpOutcome && scpType != SCP_049
            && (attackers_to(square<KING>(~sideToMove)) & pieces(sideToMove))))
        assert(0 && "pos_is_ok: Kings");

    if (pieces() & st->scpBB)
        assert(0 && "pos_is_ok: SCP-096");

    if ((pieces(PAWN) & (Rank1BB | Rank8BB)) || count<PAWN>(WHITE) > 8 || count<PAWN>(BLACK) > 8)
        assert(0 && "pos_is_ok: Pawns");


    if (ep_square() != SQ_NONE && scpType == SCP_049)
    {
        if (!((ep_square() + pawn_push(~sideToMove)) & pieces(~sideToMove, PAWN) & ~st->zombieBB)
            || !(attacks_bb(PAWN, ep_square(), ~sideToMove) & pieces(sideToMove, PAWN) & ~st->zombieBB))
            assert(0 && "pos_is_ok: En passant square (SCP-049)");
    }
    else if (ep_square() != SQ_NONE)
    {
        Square ksq = square<KING>(sideToMove);

        Bitboard captured = (ep_square() + pawn_push(~sideToMove)) & pieces(~sideToMove, PAWN);
        Bitboard pawns    = attacks_bb(PAWN, ep_square(), ~sideToMove) & pieces(sideToMove, PAWN);
        Bitboard potentialCheckers = pieces(~sideToMove) ^ captured;

        if (!captured || !pawns
            || ((attackers_to(ksq, blockers() ^ captured ^ ep_square() ^ lsb(pawns))
                 & potentialCheckers)
                && (attackers_to(ksq, blockers() ^ captured ^ ep_square() ^ msb(pawns))
                    & potentialCheckers)))
            assert(0 && "pos_is_ok: En passant square");
    }

    if ((pieces(WHITE) & pieces(BLACK)) || (pieces(WHITE) | pieces(BLACK)) != pieces()
        || popcount(pieces(WHITE)) > 16 || popcount(pieces(BLACK)) > 16)
        assert(0 && "pos_is_ok: Bitboards");

    for (PieceType p1 = PAWN; p1 <= KING; ++p1)
        for (PieceType p2 = PAWN; p2 <= KING; ++p2)
            if (p1 != p2 && (pieces(p1) & pieces(p2)))
                assert(0 && "pos_is_ok: Bitboards");

    for (Piece pc : Pieces)
        if (pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc)))
            || pieceCount[pc] != std::count(board.begin(), board.end(), pc))
            assert(0 && "pos_is_ok: Pieces");

    for (Color c : {WHITE, BLACK})
        for (CastlingRights cr : {c & KING_SIDE, c & QUEEN_SIDE})
        {
            if (!can_castle(cr))
                continue;

            if (piece_on(castling_rook_square(cr)) != make_piece(c, ROOK)
                || castlingRightsMask[castlingRookSquare[cr]] != cr
                || (castlingRightsMask[square<KING>(c)] & cr) != cr)
                assert(0 && "pos_is_ok: Castling");
        }

    assert(material_key_is_ok() && "pos_is_ok: materialKey");

    return true;
}

}  // namespace Stockfish
