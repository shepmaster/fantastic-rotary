#![feature(min_const_generics)]
#![allow(dead_code, unused_imports)]
#![deny(rust_2018_idioms)]

use jetscii::bytes;
use snafu::{ResultExt, Snafu};
use std::{
    io::{self, Read},
    ops::Deref,
    str,
};

#[derive(Debug)]
struct Exhausted(bool);

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, Hash)]
struct StringRingStats {
    /// Number of bytes of leading invalid data
    n_invalid: usize,
    /// Number of bytes of valid UTF-8 data
    n_utf8: usize,
    /// Number of bytes of read but not yet valid UTF-8 data
    n_raw: usize,
}

#[derive(Debug)]
struct StringRing<const N: usize> {
    buffer: Box<[u8; N]>,
    stats: StringRingStats,
}

impl<const N: usize> StringRing<N> {
    fn new() -> Self {
        Self {
            buffer: Box::new([0; N]),
            stats: Default::default(),
        }
    }

    fn stats(&self) -> StringRingStats {
        self.stats
    }

    fn extend(&mut self, mut rdr: impl Read) -> io::Result<Exhausted> {
        let Self { buffer, stats } = self;
        let StringRingStats {
            n_invalid,
            n_utf8,
            n_raw,
        } = stats;

        let free = &mut buffer[*n_invalid..][*n_utf8..][*n_raw..];
        assert_ne!(free.len(), 0, "todo: handle full buffer");

        let n_new_raw_bytes = rdr.read(free)?;
        if n_new_raw_bytes == 0 {
            return Ok(Exhausted(true));
        }
        *n_raw += n_new_raw_bytes;

        let raw = &buffer[*n_invalid..][*n_utf8..][..*n_raw];
        assert_ne!(raw.len(), 0, "todo: handle empty raw");

        let n_new_utf8_bytes = match str::from_utf8(raw) {
            Ok(s) => s.len(),
            Err(e) => match e.error_len() {
                None => e.valid_up_to(),
                Some(_) => todo!("Report invalid UTF-8"),
            },
        };
        *n_raw -= n_new_utf8_bytes;
        *n_utf8 += n_new_utf8_bytes;

        Ok(Exhausted(false))
    }

    fn while_tag_name(&self) -> Streaming<MatchTicket> {
        let Self { buffer, stats } = self;
        let StringRingStats { n_invalid, n_utf8, .. } = *stats;

        let utf8 = &buffer[n_invalid..][..n_utf8];

        // TODO: check valid tag name chars
        let loc = utf8.iter().position(|&b| !(b as char).is_ascii_alphabetic());

        let len = match loc {
            Some(i) => Streaming::Complete(i),
            None => Streaming::Partial(utf8.len()),
        };

        len.map(MatchTicket)
    }

    fn while_text(&self) -> Streaming<MatchTicket> {
        let Self { buffer, stats } = self;
        let StringRingStats { n_invalid, n_utf8, .. } = *stats;

        let utf8 = &buffer[n_invalid..][..n_utf8];

        let ws = bytes!('<', '&');
        let loc = ws.find(utf8);

        let len = match loc {
            Some(i) => Streaming::Complete(i),
            None => Streaming::Partial(utf8.len()),
        };

        len.map(MatchTicket)
    }

    fn start_matches(&self, needle: &str) -> MatchStatus {
        let Self { buffer, stats } = self;
        let StringRingStats { n_invalid, n_utf8, .. } = *stats;

        if n_utf8 < needle.len() {
            return MatchStatus::IncompleteInput;
        }

        let utf8 = &buffer[n_invalid..][..n_utf8];
        if utf8.starts_with(needle.as_bytes()) {
            MatchStatus::Success(MatchTicket(needle.len()))
        } else {
            MatchStatus::Failure
        }
    }

    fn exchange_ticket(&mut self, ticket: MatchTicket) -> &'_ str {
        let Self { buffer, stats } = self;
        let StringRingStats { n_invalid, n_utf8, .. } = stats;

        let utf8 = &buffer[*n_invalid..][..*n_utf8];

        let MatchTicket(len) = ticket;

        // SAFETY: Haha. I don't care yet.
        let s = unsafe {
            let b = utf8.get_unchecked(..len);
            str::from_utf8_unchecked(b)
        };

        *n_invalid += len;
        *n_utf8 -= len;

        s
    }
}

impl<const N: usize> Deref for StringRing<N> {
    type Target = str;

    fn deref(&self) -> &str {
        let Self { buffer, stats } = self;
        let StringRingStats { n_invalid, n_utf8, .. } = *stats;

        let utf8 = &buffer[n_invalid..][..n_utf8];

        // SAFETY: Haha. I don't care yet.
        unsafe { str::from_utf8_unchecked(utf8) }
    }
}

#[cfg(test)]
mod string_ring_tests {
    use super::*;
    use proptest::{collection, prelude::*, num};
    use paste::paste;

    const MAX_DATA_SIZE: usize = 1024 * 1024;

    macro_rules! const_prop {
        ($($n: literal),* $(,)?) => {
            paste! {
                proptest! {
                    $(
                        #[test]
                        fn [<doesnt_crash_ $n>](data in collection::vec(num::u8::ANY, 0..MAX_DATA_SIZE)) {
                            let mut sr: StringRing<{$n}> = StringRing::new();
                            let mut data = &data[..];
                            sr.extend(&mut data).expect("extending failed");
                        }
                    )*
                }
            }
        };
    }

    const_prop!(0, 1, 2, 3, 4, 5, 6, 7, 8);
}

#[derive(Debug)]
enum MatchStatus {
    Success(MatchTicket),
    Failure,
    IncompleteInput,
}

#[derive(Debug)]
struct MatchTicket(usize);

#[derive(Debug)]
struct Parser<R> {
    mediator: Mediator<R>,
    state: State,
}

impl<R> Parser<R>
where
    R: Read,
{
    fn new(input: R) -> Self {
        Self {
            mediator: Mediator::new(input),
            state: State::Beginning,
        }
    }

    fn next(&mut self) -> Option<Result<Token<'_>>> {
        use State::*;

        let Self { mediator, state } = self;

        // TODO: review `continue` and see if there's a better way to avoid cycling
        loop {
            match *state {
                Beginning => match mediator.starts_with("<?xml?>") {
                    Nre::Matched(s) => {
                        *state = FoundPreamble;
                        return Some(Ok(Token::Preamble(s)));
                    }
                    Nre::Error(e) => return Some(Err(e)),
                    Nre::NotMatched => {
                        todo!("Handle failure to match");
                    }
                    Nre::Exhausted => return None,
                },

                FoundPreamble => {
                    // TODO: chew whitespace
                    match mediator.starts_with("<") {
                        Nre::Matched(_) => {
                            *state = TagStart;
                            continue;
                        }
                        Nre::Error(e) => return Some(Err(e)),
                        Nre::NotMatched => {
                            todo!("Handle failure to match");
                        }
                        Nre::Exhausted => return None,
                    }
                }

                TagStart => {
                    match mediator.starts_with("/") {
                        Nre::Matched(_) => {
                            *state = InsideCloseTag;
                            return Some(Ok(Token::CloseTagStart));
                        }
                        Nre::Error(e) => return Some(Err(e)),
                        Nre::NotMatched => { /* Fall through */ }
                        Nre::Exhausted => return None,
                    }

                    *state = InsideOpenTag;
                    return Some(Ok(Token::OpenTagStart));
                }

                InsideOpenTag => {
                    let s = mediator.stream_while_tag_name();
                    if s.is_complete() {
                        *state = ReadOpenTagName;
                    }
                    return Some(Ok(Token::TagName(s)));
                }

                ReadOpenTagName => {
                    // TODO: chew whitespace

                    match mediator.starts_with(">") {
                        Nre::Matched(_) => {
                            *state = InsideElement;
                            return Some(Ok(Token::OpenTagEnd));
                        }
                        Nre::Error(e) => return Some(Err(e)),
                        Nre::NotMatched => {
                            todo!("Handle failure to match");
                        }
                        Nre::Exhausted => return None,
                    }
                }

                InsideCloseTag => {
                    let s = mediator.stream_while_tag_name();
                    if s.is_complete() {
                        *state = ReadCloseTagName;
                    }
                    return Some(Ok(Token::TagName(s)));
                }

                ReadCloseTagName => {
                    // TODO: chew whitespace

                    match mediator.starts_with(">") {
                        Nre::Matched(_) => {
                            *state = InsideElement;
                            return Some(Ok(Token::CloseTagEnd));
                        }
                        Nre::Error(e) => return Some(Err(e)),
                        Nre::NotMatched => {
                            todo!("Handle failure to match");
                        }
                        Nre::Exhausted => return None,
                    }
                }

                InsideElement => {
                    match mediator.starts_with("<") {
                        Nre::Matched(_) => {
                            *state = TagStart;
                            continue;
                        }
                        Nre::Error(e) => return Some(Err(e)),
                        Nre::NotMatched => { /* Fall through */ }
                        Nre::Exhausted => return None,
                    }

                    *state = InsideText;
                    continue;
                }

                InsideText => {
                    let s = mediator.stream_while_text();
                    if s.is_complete() {
                        *state = InsideElement;
                    }
                    return Some(Ok(Token::Text(s)));
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum State {
    Beginning,
    FoundPreamble,
    TagStart,
    InsideOpenTag,
    ReadOpenTagName,
    InsideCloseTag,
    ReadCloseTagName,
    InsideElement,
    InsideText,
}

#[derive(Debug)]
struct Mediator<R> {
    input: R,
    buffer: StringRing<1024>,
}

impl<R> Mediator<R>
where
    R: io::Read,
{
    fn new(input: R) -> Self {
        Self {
            input,
            buffer: StringRing::new(),
        }
    }

    fn as_str(&self) -> &str {
        &self.buffer
    }

    fn starts_with(&mut self, ss: &str) -> Nre<&'_ str, Error> {
        loop {
            let s = self.buffer.stats();
            match self.buffer.start_matches(ss) {
                MatchStatus::Success(ticket) => {
                    let s = self.buffer.exchange_ticket(ticket);
                    return Nre::Matched(s);
                }
                MatchStatus::Failure => {
                    return Nre::NotMatched;
                }
                MatchStatus::IncompleteInput => match self.buffer.extend(&mut self.input).context(UnableToReadData) {
                    Ok(Exhausted(true)) => return Nre::Exhausted,
                    Ok(Exhausted(false)) => {}
                    Err(e) => return Nre::Error(e),
                },
            }

            assert_ne!(s, self.buffer.stats(), "Stats did not change ({:?})", s);
        }
    }

    fn stream_while_tag_name(&mut self) -> Streaming<&'_ str> {
        self.buffer
            .while_tag_name()
            .map(move |ticket| self.buffer.exchange_ticket(ticket))
    }

    fn stream_while_text(&mut self) -> Streaming<&'_ str> {
        self.buffer
            .while_text()
            .map(move |ticket| self.buffer.exchange_ticket(ticket))
    }
}

// TODO: unify with MatchStatus?
#[derive(Debug)]
enum Nre<T, E> {
    Exhausted,
    Matched(T),
    NotMatched,
    Error(E),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum Streaming<T> {
    Partial(T),
    Complete(T),
}

impl<T> Streaming<T> {
    fn is_partial(&self) -> bool {
        matches!(self, Streaming::Partial(_))
    }

    fn is_complete(&self) -> bool {
        matches!(self, Streaming::Complete(_))
    }

    fn as_ref(&self) -> Streaming<&T> {
        use Streaming::*;

        match self {
            Partial(v) => Partial(v),
            Complete(v) => Complete(v),
        }
    }

    fn map<U>(self, f: impl FnOnce(T) -> U) -> Streaming<U> {
        use Streaming::*;

        match self {
            Partial(v) => Partial(f(v)),
            Complete(v) => Complete(f(v)),
        }
    }

    fn unify(self) -> T {
        use Streaming::*;

        match self {
            Partial(v) => v,
            Complete(v) => v,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
enum Token<'a> {
    Preamble(&'a str),
    OpenTagStart,
    OpenTagEnd,
    CloseTagStart,
    CloseTagEnd,
    TagName(Streaming<&'a str>),
    Text(Streaming<&'a str>),
}

impl<'a> Token<'a> {
    fn to_owned(self) -> OwnedToken {
        use OwnedToken as OT;
        use Token as T;

        match self {
            T::Preamble(s) => OT::Preamble(s.to_owned()),
            T::OpenTagStart => OT::OpenTagStart,
            T::OpenTagEnd => OT::OpenTagEnd,
            T::CloseTagStart => OT::CloseTagStart,
            T::CloseTagEnd => OT::CloseTagEnd,
            T::TagName(s) => OT::TagName(s.map(|s| s.to_owned())),
            T::Text(s) => OT::Text(s.map(|s| s.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum OwnedToken {
    Preamble(String),
    OpenTagStart,
    OpenTagEnd,
    CloseTagStart,
    CloseTagEnd,
    TagName(Streaming<String>),
    Text(Streaming<String>),
}

#[derive(Debug, Snafu)]
enum Error {
    UnableToReadData { source: io::Error },
    InvalidUtf8,
}

type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_basic_document() {
        use OwnedToken::*;
        use Streaming::*;

        let input = b"<?xml?><alpha>hello</alpha>";

        let mut p = Parser::new(&input[..]);
        let mut x = vec![];
        while let Some(v) = p.next() {
            let a = v.unwrap().to_owned();
            x.push(a);
        }

        assert_eq!(
            x,
            [
                Preamble("<?xml?>".into()),
                OpenTagStart,
                TagName(Complete("alpha".into())),
                OpenTagEnd,
                Text(Complete("hello".into())),
                CloseTagStart,
                TagName(Complete("alpha".into())),
                CloseTagEnd,
            ]
        );
    }
}
