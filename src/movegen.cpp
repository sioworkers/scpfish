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

#include "movegen.h"

#include <cassert>
#include <initializer_list>

#include "attacks.h"
#include "bitboard.h"
#include "position.h"

#if defined(USE_AVX512ICL)
    #include <array>
    #include <algorithm>
    #include <immintrin.h>
#endif

namespace Stockfish {

namespace {

#if defined(USE_AVX512ICL)

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    assert(popcount(to_bb) <= 8);  // <= 8 pawns per side

    const __m128i toSquares =
      _mm_cvtepi8_epi16(_mm512_castsi512_si128(_mm512_maskz_compress_epi8(to_bb, AllSquares)));
    const __m128i fromSquares = _mm_subs_epi16(toSquares, _mm_set1_epi16(offset));
    const __m128i moves       = _mm_or_si128(_mm_slli_epi16(fromSquares, Move::FromSqShift),
                                             _mm_slli_epi16(toSquares, Move::ToSqShift));

    _mm_storeu_si128(reinterpret_cast<__m128i*>(moveList), moves);
    return moveList + popcount(to_bb);
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    assert(popcount(to_bb) <= 32);  // Q can attack up to 27 squares

    const __m512i fromVec = _mm512_set1_epi16(Move(from, SQUARE_ZERO).raw());
    const __m512i toSquares =
      _mm512_cvtepi8_epi16(_mm512_castsi512_si256(_mm512_maskz_compress_epi8(to_bb, AllSquares)));
    const __m512i moves = _mm512_or_si512(fromVec, _mm512_slli_epi16(toSquares, Move::ToSqShift));

    _mm512_storeu_si512(moveList, moves);
    return moveList + popcount(to_bb);
}

#else

template<Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    while (to_bb)
    {
        Square to   = pop_lsb(to_bb);
        *moveList++ = Move(to - offset, to);
    }
    return moveList;
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    while (to_bb)
        *moveList++ = Move(from, pop_lsb(to_bb));
    return moveList;
}

#endif

template<GenType Type, Direction D, bool Enemy>
Move* make_promotions(Move* moveList, [[maybe_unused]] Square to) {

    constexpr bool          all  = Type == EVASIONS || Type == ALL;
    [[maybe_unused]] Square from = to - D;

    if constexpr (Type == CAPTURES || all)
        *moveList++ = Move::make<PROMOTION>(from, to, QUEEN);

    if constexpr ((Type == CAPTURES && Enemy) || (Type == QUIETS && !Enemy) || all)
    {
        *moveList++ = Move::make<PROMOTION>(from, to, ROOK);
        *moveList++ = Move::make<PROMOTION>(from, to, BISHOP);
        *moveList++ = Move::make<PROMOTION>(from, to, KNIGHT);
    }

    return moveList;
}


template<Color Us, GenType Type>
Move* generate_pawn_moves(const Position& pos, Move* moveList, Bitboard target) {

    static_assert(Type != ALL, "Unsupported type in generate_pawn_moves()");

    constexpr Color     Them     = ~Us;
    constexpr Bitboard  TRank7BB = (Us == WHITE ? Rank7BB : Rank2BB);
    constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up       = pawn_push(Us);
    constexpr Direction UpRight  = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft   = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~pos.blockers();
    const Bitboard enemies =
      Type == EVASIONS ? pos.checkers() & pos.pieces(Them) : pos.pieces(Them);
    const Bitboard pinned       = pos.blockers_for_king(Us);
    const Square   ksq          = pos.square<KING>(Us);

    const Bitboard pawnsOn7        = pos.pieces(Us, PAWN) & TRank7BB;
    const Bitboard pawnsNotOn7     = pos.pieces(Us, PAWN) & ~TRank7BB;
    const Bitboard pushable        = ~pinned | file_bb(ksq);
    const Bitboard canCaptureLeft  = ~pinned | Attacks::antidiag_bb(ksq);
    const Bitboard canCaptureRight = ~pinned | Attacks::diag_bb(ksq);

    // Single and double pawn pushes, no promotions
    if constexpr (Type != CAPTURES)
    {
        Bitboard b1 = shift(pawnsNotOn7 & pushable, Up) & emptySquares;
        Bitboard b2 = shift(b1 & TRank3BB, Up) & emptySquares;

        if constexpr (Type == EVASIONS)  // Consider only blocking squares
        {
            b1 &= target;
            b2 &= target;
        }

        moveList = splat_pawn_moves<Up>(moveList, b1);
        moveList = splat_pawn_moves<Up + Up>(moveList, b2);
    }

    // Promotions and underpromotions
    if (pawnsOn7)
    {
        Bitboard b1 = shift(pawnsOn7 & canCaptureRight, UpRight) & enemies;
        Bitboard b2 = shift(pawnsOn7 & canCaptureLeft, UpLeft) & enemies;
        Bitboard b3 = shift(pawnsOn7 & pushable, Up) & emptySquares;

        if constexpr (Type == EVASIONS)
            b3 &= target;

        while (b1)
            moveList = make_promotions<Type, UpRight, true>(moveList, pop_lsb(b1));

        while (b2)
            moveList = make_promotions<Type, UpLeft, true>(moveList, pop_lsb(b2));

        while (b3)
            moveList = make_promotions<Type, Up, false>(moveList, pop_lsb(b3));
    }

    // Standard and en passant captures
    if constexpr (Type == CAPTURES || Type == EVASIONS)
    {
        Bitboard b1 = shift(pawnsNotOn7 & canCaptureRight, UpRight) & enemies;
        Bitboard b2 = shift(pawnsNotOn7 & canCaptureLeft, UpLeft) & enemies;

        moveList = splat_pawn_moves<UpRight>(moveList, b1);
        moveList = splat_pawn_moves<UpLeft>(moveList, b2);

        const Square epSq = pos.ep_square();

        if (epSq != SQ_NONE
            // An en passant capture cannot resolve a discovered check
            && !(Type == EVASIONS && (target & (epSq + Up))))
        {
            assert(rank_of(epSq) == relative_rank(Us, RANK_6));

            b1 = pawnsNotOn7
               & ((canCaptureRight & shift(square_bb(epSq), -UpRight))
                  | (canCaptureLeft & shift(square_bb(epSq), -UpLeft)));
            assert(b1);

            while (b1)
                *moveList++ = Move::make<EN_PASSANT>(pop_lsb(b1), epSq);
        }
    }

    return moveList;
}


template<PieceType Pt>
Move* generate_moves(const Position& pos, Move* moveList, Bitboard target, Color us) {

    static_assert(Pt != KING && Pt != PAWN, "Unsupported piece type in generate_moves()");

    const Square   ksq    = pos.square<KING>(us);
    const Bitboard pinned = pos.blockers_for_king(us);

    Bitboard bb = pos.pieces(us, Pt) & ~pinned;
    while (bb)
    {
        Square   from = pop_lsb(bb);
        Bitboard b    = Attacks::attacks_bb(Pt, from, pos.blockers()) & target;

        moveList = splat_moves(moveList, from, b);
    }

    if constexpr (Pt != KNIGHT)  // pinned knights cannot move
    {
        bb = pos.pieces(us, Pt) & pinned;
        while (bb)
        {
            Square   from = pop_lsb(bb);
            Bitboard b    = Attacks::attacks_bb(Pt, from, pos.blockers()) & target;
            b &= Attacks::line_bb(ksq, from);

            moveList = splat_moves(moveList, from, b);
        }
    }

    return moveList;
}


template<GenType Type>
Move* generate_all(const Position& pos, Move* moveList, Color us) {

    static_assert(Type != ALL, "Unsupported type in generate_all()");

    const Square ksq = pos.square<KING>(us);
    Bitboard     target;

    // Skip generating non-king moves when in double check
    if (Type != EVASIONS || !more_than_one(pos.checkers()))
    {
        // An SCP-096 check can only be evaded by a king move
        target = Type == EVASIONS ? Attacks::between_bb(ksq, lsb(pos.checkers())) & ~pos.scp_bb()
               : Type == CAPTURES ? pos.pieces(~us)
                                  : ~pos.blockers();  // QUIETS

        moveList = us == WHITE ? generate_pawn_moves<WHITE, Type>(pos, moveList, target)
                               : generate_pawn_moves<BLACK, Type>(pos, moveList, target);
        moveList = generate_moves<KNIGHT>(pos, moveList, target, us);
        moveList = generate_moves<BISHOP>(pos, moveList, target, us);
        moveList = generate_moves<ROOK>(pos, moveList, target, us);
        moveList = generate_moves<QUEEN>(pos, moveList, target, us);
    }

    if constexpr (Type == EVASIONS)
        target = ~pos.pieces(us);

    // Kings may not stand on or a knight's move from SCP-096
    target &= ~(pos.threats_by(ALL_PIECES) | pos.scp_king_forbidden());

    Bitboard b = Attacks::attacks_bb(KING, ksq) & target;
    moveList   = splat_moves(moveList, ksq, b);

    // Castling is allowed when SCP-096 is the only checker
    if ((Type == QUIETS || (Type == EVASIONS && pos.checkers() == pos.scp_bb()))
        && pos.can_castle(us & ANY_CASTLING))
    {
        for (CastlingRights cr : {us & KING_SIDE, us & QUEEN_SIDE})
        {
            Square rookSquare = pos.castling_rook_square(cr);
            Square to         = relative_square(us, rookSquare > ksq ? SQ_G1 : SQ_C1);

            if (!pos.castling_impeded(cr) && pos.can_castle(cr)
                && !(Attacks::between_bb(ksq, to) & pos.threats_by(ALL_PIECES))
                && !(pos.scp_king_forbidden() & to)
                && !(pos.blockers_for_king(us) & rookSquare))
                *moveList++ = Move::make<CASTLING>(ksq, rookSquare);
        }
    }

    return moveList;
}


// Generates pseudo-legal moves under SCP-049 rules. There is no check, zombies
// neither move nor can be captured, and king captures are not generated (a
// position where one is available is scored as won). EVASIONS generates all moves.
template<GenType Type>
Move* generate_049(const Position& pos, Move* moveList) {

    constexpr bool Caps   = Type == CAPTURES || Type == EVASIONS;
    constexpr bool Quiets = Type == QUIETS || Type == EVASIONS;

    const Color    us      = pos.side_to_move();
    const Color    them    = ~us;
    const Bitboard zombies = pos.zombies();
    const Bitboard occ     = pos.blockers();
    const Bitboard enemies = pos.pieces(them) & ~zombies & ~pos.pieces(KING);
    const Bitboard emptySq = ~occ;
    const Bitboard target  = (Caps ? enemies : Bitboard(0)) | (Quiets ? emptySq : Bitboard(0));

    // Pawns
    const Direction up     = pawn_push(us);
    const Bitboard  rank8  = us == WHITE ? Rank8BB : Rank1BB;
    const Bitboard  rank3  = us == WHITE ? Rank3BB : Rank6BB;
    const Bitboard  pawns  = pos.pieces(us, PAWN) & ~zombies;

    auto promotions = [&](Square from, Square to, bool capture) {
        if (Caps)
            *moveList++ = Move::make<PROMOTION>(from, to, QUEEN);
        if ((Type == CAPTURES && capture) || (Type == QUIETS && !capture) || Type == EVASIONS)
            for (PieceType pt : {ROOK, BISHOP, KNIGHT})
                *moveList++ = Move::make<PROMOTION>(from, to, pt);
    };

    const Bitboard push1 = shift(pawns, up) & emptySq;
    const Bitboard push2 = shift(push1 & rank3, up) & emptySq;

    for (Bitboard b = push1; b;)
    {
        const Square to = pop_lsb(b);
        if (rank8 & to)
            promotions(to - up, to, false);
        else if (Quiets)
            *moveList++ = Move(to - up, to);
    }
    if (Quiets)
        for (Bitboard b = push2; b;)
        {
            const Square to = pop_lsb(b);
            *moveList++     = Move(to - up - up, to);
        }

    for (Bitboard b = pawns; b;)
    {
        const Square from = pop_lsb(b);
        Bitboard     caps = Attacks::attacks_bb(PAWN, from, us) & enemies;
        while (caps)
        {
            const Square to = pop_lsb(caps);
            if (rank8 & to)
                promotions(from, to, true);
            else if (Caps)
                *moveList++ = Move(from, to);
        }
    }

    if (Caps && pos.ep_square() != SQ_NONE)
        for (Bitboard b = Attacks::attacks_bb(PAWN, pos.ep_square(), them) & pawns; b;)
            *moveList++ = Move::make<EN_PASSANT>(pop_lsb(b), pos.ep_square());

    // Pieces
    for (Bitboard b = pos.pieces(us) & ~zombies & ~pos.pieces(PAWN); b;)
    {
        const Square    from = pop_lsb(b);
        const PieceType pt   = type_of(pos.piece_on(from));
        moveList = splat_moves(moveList, from, Attacks::attacks_bb(pt, from, occ) & target);
    }

    // Castling as on the site: no piece between king and rook (the SCP does not
    // count), the SCP not on the king's destination, no king square attacked.
    const Square ksq = pos.square<KING>(us);
    if (Quiets && !(zombies & ksq) && pos.can_castle(us & ANY_CASTLING) && !pos.scp049_attacked(ksq, them))
        for (CastlingRights cr : {us & KING_SIDE, us & QUEEN_SIDE})
        {
            if (!pos.can_castle(cr) || (pos.pieces() & pos.castling_path(cr)))
                continue;

            const Square rsq = pos.castling_rook_square(cr);
            const Square kto = relative_square(us, rsq > ksq ? SQ_G1 : SQ_C1);
            const Square mid = relative_square(us, rsq > ksq ? SQ_F1 : SQ_D1);

            if (!(pos.scp_bb() & kto) && !pos.scp049_attacked(mid, them)
                && !pos.scp049_attacked(kto, them))
                *moveList++ = Move::make<CASTLING>(ksq, rsq);
        }

    return moveList;
}

}  // namespace


// <CAPTURES>     Generates all legal captures plus queen promotions
// <QUIETS>       Generates all legal non-captures and underpromotions
// <EVASIONS>     Generates all legal check evasions
//
// Returns a pointer to the end of the move list.
template<GenType Type>
Move* generate(const Position& pos, Move* moveList) {

    static_assert(Type != ALL, "Unsupported type in generate()");
    assert((Type == EVASIONS) == bool(pos.checkers()));

    if (pos.scp_type() == SCP_049)
        return generate_049<Type>(pos, moveList);

    Color us = pos.side_to_move();

    return generate_all<Type>(pos, moveList, us);
}

// Explicit template instantiations
template Move* generate<CAPTURES>(const Position&, Move*);
template Move* generate<QUIETS>(const Position&, Move*);
template Move* generate<EVASIONS>(const Position&, Move*);

// generate<ALL> generates all the legal moves in the given position

template<>
Move* generate<ALL>(const Position& pos, Move* moveList) {
    return pos.checkers() ? generate<EVASIONS>(pos, moveList)
                          : generate<QUIETS>(pos, generate<CAPTURES>(pos, moveList));
}

}  // namespace Stockfish
